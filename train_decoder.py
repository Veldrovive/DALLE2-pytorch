from dalle2_pytorch import Unet, Decoder
from dalle2_pytorch.trainer import DecoderTrainer, print_ribbon
from dalle2_pytorch.dataloaders import create_image_embedding_dataloader
from dalle2_pytorch.trackers import WandbTracker, ConsoleTracker
from configs.decoder_defaults import default_config, ConfigField
import time
import os
import json
import torchvision
from torchvision import transforms as T
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import reduce
import webdataset as wds
import click


def create_dataloaders(
    available_shards,
    webdataset_base_url,
    embeddings_url,
    shard_width=6,
    num_workers=4,
    batch_size=32,
    n_sample_images=6,
    shuffle_train=True,
    resample_train=False,
    img_preproc = None,
    index_width=4,
    train_prop = 0.75,
    val_prop = 0.15,
    test_prop = 0.10,
    **kwargs
):
    """
    Randomly splits the available shards into train, val, and test sets and returns a dataloader for each
    """
    assert train_prop + test_prop + val_prop == 1
    num_train = round(train_prop*len(available_shards))
    num_test = round(test_prop*len(available_shards))
    num_val = len(available_shards) - num_train - num_test
    assert num_train + num_test + num_val == len(available_shards), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(available_shards)}"
    train_split, test_split, val_split = torch.utils.data.random_split(available_shards, [num_train, num_test, num_val], generator=torch.Generator().manual_seed(0))

    # The shard number in the webdataset file names has a fixed width. We zero pad the shard numbers so they correspond to a filename.
    train_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in train_split]
    test_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in test_split]
    val_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in val_split]
    
    create_dataloader = lambda tar_urls, shuffle=False, resample=False, with_text=False, for_sampling=False: create_image_embedding_dataloader(
        tar_url=tar_urls,
        num_workers=num_workers,
        batch_size=batch_size if not for_sampling else n_sample_images,
        embeddings_url=embeddings_url,
        index_width=index_width,
        shuffle_num = None,
        extra_keys= ["txt"] if with_text else [],
        shuffle_shards = shuffle,
        resample_shards = resample, 
        img_preproc=img_preproc,
        handler=wds.handlers.warn_and_continue
    )

    train_dataloader = create_dataloader(train_urls, shuffle=shuffle_train, resample=resample_train)
    train_sampling_dataloader = create_dataloader(train_urls, shuffle=False, for_sampling=True)
    val_dataloader = create_dataloader(val_urls, shuffle=False, with_text=True)
    test_dataloader = create_dataloader(test_urls, shuffle=False, with_text=True)
    test_sampling_dataloader = create_dataloader(test_urls, shuffle=False, for_sampling=True)
    return {
        "train": train_dataloader,
        "train_sampling": train_sampling_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
        "test_sampling": test_sampling_dataloader
    }


def create_decoder(device, decoder_config, unets_config):
    """Creates a sample decoder"""
    unets = []
    for i in range(0, len(unets_config)):
        unets.append(Unet(
            **unets_config[i]
        ))
    
    decoder = Decoder(
        unet=tuple(unets),  # Must be tuple because of cast_tuple
        **decoder_config
    )
    decoder.to(device=device)

    return decoder

def get_dataset_keys(dataloader):
    """
    It is sometimes neccesary to get the keys the dataloader is returning. Since the dataset is burried in the dataloader, we need to do a process to recover it.
    """
    # If the dataloader is actually a WebLoader, we need to extract the real dataloader
    if isinstance(dataloader, wds.WebLoader):
        dataloader = dataloader.pipeline[0]
    return dataloader.dataset.key_map

def get_example_data(dataloader, device, n=5):
    """
    Samples the dataloader and returns a zipped list of examples
    """
    images = []
    embeddings = []
    captions = []
    dataset_keys = get_dataset_keys(dataloader)
    has_caption = "txt" in dataset_keys
    for data in dataloader:
        if has_caption:
            img, emb, txt = data
        else:
            img, emb = data
            txt = [""] * emb.shape[0]
        img = img.to(device=device, dtype=torch.float)
        emb = emb.to(device=device, dtype=torch.float)
        images.extend(list(img))
        embeddings.extend(list(emb))
        captions.extend(list(txt))
        if len(images) >= n:
            break
    print("Generated {} examples".format(len(images)))
    return list(zip(images[:n], embeddings[:n], captions[:n]))

def generate_samples(trainer, example_data, text_prepend=""):
    """
    Takes example data and generates images from the embeddings
    Returns three lists: real images, generated images, and captions
    """
    real_images, embeddings, txts = zip(*example_data)
    embeddings_tensor = torch.stack(embeddings)
    samples = trainer.sample(embeddings_tensor)
    generated_images = list(samples)
    captions = [text_prepend + txt for txt in txts]
    return real_images, generated_images, captions

def generate_grid_samples(trainer, examples, text_prepend=""):
    """
    Generates samples and uses torchvision to put them in a side by side grid for easy viewing
    """
    real_images, generated_images, captions = generate_samples(trainer, examples, text_prepend)
    grid_images = [torchvision.utils.make_grid([original_image, generated_image]) for original_image, generated_image in zip(real_images, generated_images)]
    return grid_images, captions
                    
def evaluate_trainer(trainer, dataloader, device, n_evalation_samples=1000, FID=None, IS=None, KID=None, LPIPS=None):
    """
    Computes evaluation metrics for the decoder
    """
    metrics = {}
    # Prepare the data
    examples = get_example_data(dataloader, device, n_evalation_samples)
    real_images, generated_images, captions = generate_samples(trainer, examples)
    real_images = torch.stack(real_images).to(device=device, dtype=torch.float)
    generated_images = torch.stack(generated_images).to(device=device, dtype=torch.float)
    # Convert from [0, 1] to [0, 255] and from torch.float to torch.uint8
    int_real_images = real_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
    int_generated_images = generated_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
    if FID is not None:
        fid = FrechetInceptionDistance(**FID)
        fid.to(device=device)
        fid.update(int_real_images, real=True)
        fid.update(int_generated_images, real=False)
        metrics["FID"] = fid.compute().item()
    if IS is not None:
        inception = InceptionScore(**IS)
        inception.to(device=device)
        inception.update(int_real_images)
        is_mean, is_std = inception.compute()
        metrics["IS_mean"] = is_mean.item()
        metrics["IS_std"] = is_std.item()
    if KID is not None:
        kernel_inception = KernelInceptionDistance(**KID)
        kernel_inception.to(device=device)
        kernel_inception.update(int_real_images, real=True)
        kernel_inception.update(int_generated_images, real=False)
        kid_mean, kid_std = kernel_inception.compute()
        metrics["KID_mean"] = kid_mean.item()
        metrics["KID_std"] = kid_std.item()
    if LPIPS is not None:
        # Convert from [0, 1] to [-1, 1]
        renorm_real_images = real_images.mul(2).sub(1)
        renorm_generated_images = generated_images.mul(2).sub(1)
        lpips = LearnedPerceptualImagePatchSimilarity(**LPIPS)
        lpips.to(device=device)
        lpips.update(renorm_real_images, renorm_generated_images)
        metrics["LPIPS"] = lpips.compute().item()
    return metrics

def save_trainer(tracker, trainer, epoch, step, validation_losses, relative_paths):
    """
    Logs the model with an appropriate method depending on the tracker
    """
    if isinstance(relative_paths, str):
        relative_paths = [relative_paths]
    trainer_state_dict = {}
    trainer_state_dict["trainer"] = trainer.state_dict()
    trainer_state_dict['epoch'] = epoch
    trainer_state_dict['step'] = step
    trainer_state_dict['validation_losses'] = validation_losses
    for relative_path in relative_paths:
        tracker.save_state_dict(trainer_state_dict, relative_path)
    
def recall_trainer(tracker, trainer, recall_source=None, **load_config):
    """
    Loads the model with an appropriate method depending on the tracker
    """
    print(print_ribbon(f"Loading model from {recall_source}"))
    state_dict = tracker.recall_state_dict(recall_source, **load_config)
    trainer.load_state_dict(state_dict["trainer"])
    print("Model loaded")
    return state_dict["epoch"], state_dict["step"], state_dict["validation_losses"]

def train(
    dataloaders,
    decoder,
    accelerator,
    tracker,
    inference_device,
    load_config=None,
    evaluate_config=None,
    epoch_samples = None,  # If the training dataset is resampling, we have to manually stop an epoch
    validation_samples = None,
    epochs = 20,
    n_sample_images = 5,
    save_every_n_samples = 100000,
    save_all=False,
    save_latest=True,
    save_best=True,
    unet_training_mask=None,
    **kwargs
):
    """
    Trains a decoder on a dataset.
    """
    is_master = accelerator.process_index == 0

    trainer = DecoderTrainer(  # TODO: Change the get_optimizer function so that it can take arbitrary named args so we can just put **kwargs as an argument here
        accelerator,
        decoder,
        **kwargs
    )

    # Set up starting model and parameters based on a recalled state dict
    start_step = 0
    start_epoch = 0
    validation_losses = []
    log_interval = 20

    if load_config is not None and load_config["source"] is not None:
        start_epoch, start_step, validation_losses = recall_trainer(tracker, trainer, recall_source=load_config["source"], **load_config)
    trainer.to(device=inference_device)

    if unet_training_mask is None:
        # Then the unet mask should be true for all unets in the decoder
        unet_training_mask = [True] * trainer.num_unets
    assert len(unet_training_mask) == trainer.num_unets, f"The unet training mask should be the same length as the number of unets in the decoder. Got {len(unet_training_mask)} and {trainer.num_unets}"

    accelerator.print(print_ribbon("Generating Example Data", repeat=40))
    accelerator.print("This can take a while to load the shard lists...")
    if is_master:
        train_example_data = get_example_data(dataloaders["train_sampling"], inference_device, n_sample_images)
        test_example_data = get_example_data(dataloaders["test_sampling"], inference_device, n_sample_images)
    
    send_to_device = lambda arr: [x.to(device=inference_device, dtype=torch.float) for x in arr]
    step = start_step

    sample_length_tensor = torch.zeros(1, dtype=torch.int, device=inference_device)
    unet_losses_tensor = torch.zeros(log_interval, trainer.num_unets, dtype=torch.float, device=inference_device)
    for epoch in range(start_epoch, epochs):
        accelerator.print(print_ribbon(f"Starting epoch {epoch}", repeat=40))
        trainer.train()

        sample = 0
        last_sample = 0
        last_snapshot = 0
        last_time = time.time()
        for i, (img, emb) in enumerate(dataloaders["train"]):
            # We want to count the total number of samples across all processes
            sample_length_tensor[0] = len(img)
            all_samples = accelerator.gather(sample_length_tensor)  # TODO: accelerator.reduce is broken when this was written. If it is fixed replace this.
            total_samples = all_samples.sum().item()
            step += 1
            sample += total_samples
            img, emb = send_to_device((img, emb))
            
            for unet in range(1, trainer.num_unets+1):
                # Check if this is a unet we are training
                if unet_training_mask[unet-1]: # Unet index is the unet number - 1
                    loss = trainer.forward(img, image_embed=emb, unet_number=unet)
                    trainer.update(unet_number=unet)
                    unet_losses_tensor[i % log_interval, unet-1] = loss

            samples_per_sec = (sample - last_sample) / (time.time() - last_time)
            last_time = time.time()
            last_sample = sample

            if i % log_interval == 0:
                # We want to average losses across all processes
                unet_all_losses = accelerator.gather(unet_losses_tensor)
                mask = unet_all_losses != 0
                unet_average_loss = (unet_all_losses * mask).sum(dim=0) / mask.sum(dim=0)
                loss_map = { f"Unet {index} Training Loss": loss.item() for index, loss in enumerate(unet_average_loss) if loss != 0 }
                log_data = {
                    "Epoch": epoch,
                    "Sample": sample,
                    "Step": i,
                    "Samples per second": samples_per_sec,
                    **loss_map
                }
                # print(f"I am rank {accelerator.state.process_index}. Example weight: {trainer.decoder.state_dict()['module.unets.0.init_conv.convs.0.weight'][0,0,0,0]}")
                if is_master:
                    tracker.log(log_data, step=step, verbose=True)


            if is_master and last_snapshot + save_every_n_samples < sample:  # This will miss by some amount every time, but it's not a big deal... I hope
                # It is difficult to gather this kind of info on the accelerator, so we have to do it on the master
                print("Saving snapshot")
                last_snapshot = sample
                # We need to know where the model should be saved
                save_paths = []
                if save_latest:
                    save_paths.append("latest.pth")
                if save_all:
                    save_paths.append(f"checkpoints/epoch_{epoch}_step_{step}.pth")
                save_trainer(tracker, trainer, epoch, step, validation_losses, save_paths)
                if n_sample_images is not None and n_sample_images > 0:
                    trainer.eval()
                    train_images, train_captions = generate_grid_samples(trainer, train_example_data, "Train: ")
                    trainer.train()
                    tracker.log_images(train_images, captions=train_captions, image_section="Train Samples", step=step)

            if epoch_samples is not None and sample >= epoch_samples:
                break

        trainer.eval()
        accelerator.print(print_ribbon(f"Starting Validation {epoch}", repeat=40))
        val_sample = 0
        val_sample_length_tensor = torch.zeros(1, dtype=torch.int, device=inference_device)
        average_val_loss_tensor = torch.zeros(1, trainer.num_unets, dtype=torch.float, device=inference_device)
        last_time = time.time()
        last_sample = 0
        for i, (img, emb, txt) in enumerate(dataloaders["val"]):
            val_sample_length_tensor[0] = len(img)
            all_samples = accelerator.gather(val_sample_length_tensor)
            total_samples = all_samples.sum().item()
            val_sample += total_samples
            img, emb = send_to_device((img, emb))
            
            for unet in range(1, len(decoder.unets)+1):
                if unet_training_mask[unet-1]:
                    loss = trainer.forward(img.float(), image_embed=emb.float(), unet_number=unet)
                    average_val_loss_tensor[0, unet-1] += loss

            samples_per_sec = (val_sample - last_sample) / (time.time() - last_time)
            last_time = time.time()
            last_sample = val_sample

            if i % log_interval == 0:
                accelerator.print(f"Epoch {epoch}/{epochs} Val Step {i} -  Sample {val_sample} - {samples_per_sec:.2f} samples/sec")
                accelerator.print(f"Loss: {(average_val_loss_tensor / (i+1))}")
                accelerator.print("")

            if validation_samples is not None and val_sample >= validation_samples:
                break
        average_val_loss_tensor /= i+1
        # Gather all the average loss tensors
        all_average_val_losses = accelerator.gather(average_val_loss_tensor)
        if is_master:
            unet_average_val_loss = all_average_val_losses.mean(dim=0)
            val_loss_map = { f"Unet {index} Validation Loss": loss.item() for index, loss in enumerate(unet_average_val_loss) if loss != 0 }
            tracker.log(val_loss_map, step=step, verbose=True)

        # Compute evaluation metrics
        # We want to do this in a distributed manner so that we can have as many samples as possible since this is slow
        trainer.eval()
        if is_master and evaluate_config is not None:
            print(print_ribbon(f"Starting Evaluation {epoch}", repeat=40))
            evaluation = evaluate_trainer(trainer, dataloaders["val"], inference_device, **evaluate_config)
            tracker.log(evaluation, step=step, verbose=True)

        if is_master:
            # Only generate examples and save the model if we are the master
            # Generate sample images
            print(print_ribbon(f"Sampling Set {epoch}", repeat=40))
            test_images, test_captions = generate_grid_samples(trainer, test_example_data, "Test: ")
            train_images, train_captions = generate_grid_samples(trainer, train_example_data, "Train: ")
            tracker.log_images(test_images, captions=test_captions, image_section="Test Samples", step=step)
            tracker.log_images(train_images, captions=train_captions, image_section="Train Samples", step=step)

            print(print_ribbon(f"Starting Saving {epoch}", repeat=40))
            # Get the same paths
            save_paths = []
            if save_latest:
                save_paths.append("latest.pth")
            average_loss = all_average_val_losses.mean(dim=0).item()
            if save_best and (len(validation_losses) == 0 or average_loss < min(validation_losses)):
                save_paths.append("best.pth")
            validation_losses.append(average_loss)
            save_trainer(tracker, trainer, epoch, step, validation_losses, save_paths)

def create_tracker(config, tracker_type=None, data_path=None, **kwargs):
    """
    Creates a tracker of the specified type and initializes special features based on the full config
    """
    tracker_config = config["tracker"]
    init_config = {}
    init_config["config"] = config.config
    if tracker_type == "console":
        tracker = ConsoleTracker(**init_config)
    elif tracker_type == "wandb":
        # We need to initialize the resume state here
        load_config = config["load"]
        if load_config["source"] == "wandb" and load_config["resume"]:
            # Then we are resuming the run load_config["run_path"]
            run_id = config["resume"]["wandb_run_path"].split("/")[-1]
            init_config["id"] = run_id
            init_config["resume"] = "must"
        init_config["entity"] = tracker_config["wandb_entity"]
        init_config["project"] = tracker_config["wandb_project"]
        tracker = WandbTracker(data_path)
        tracker.init(**init_config)
    else:
        raise ValueError(f"Tracker type {tracker_type} not supported by decoder trainer")
    return tracker
    
def initialize_training(config):
    torch.manual_seed(config["seed"] if "seed" in config.config else 0)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    all_shards = list(range(config["data"]["start_shard"], config["data"]["end_shard"] + 1))
    world_size = accelerator.state.num_processes
    rank = accelerator.state.process_index
    shards_per_process = len(all_shards) // world_size
    assert shards_per_process > 0, "Not enough shards to split evenly"
    my_shards = all_shards[rank * shards_per_process: (rank + 1) * shards_per_process]


    dataloaders = create_dataloaders (
        available_shards=my_shards,
        img_preproc = config.get_preprocessing(),
        train_prop = config["data"]["splits"]["train"],
        val_prop = config["data"]["splits"]["val"],
        test_prop = config["data"]["splits"]["test"],
        n_sample_images=config["train"]["n_sample_images"],
        **config["data"]
    )

    decoder = create_decoder(accelerator.state.device, config["decoder"], config["unets"])
    num_parameters = sum(p.numel() for p in decoder.parameters())
    accelerator.print(print_ribbon("Loaded Config", repeat=40))
    accelerator.print(f"Number of parameters: {num_parameters}")

    tracker = create_tracker(config, **config["tracker"]) if rank == 0 else None

    train(dataloaders, decoder, accelerator,
        tracker=tracker,
        inference_device=accelerator.state.device,
        load_config=config["load"],
        evaluate_config=config["evaluate"],
        **config["train"],
    )


class TrainDecoderConfig:
    def __init__(self, config):
        self.config = self.map_config(config, default_config)

    def map_config(self, config, defaults):
        """
        Returns a dictionary containing all config options in the union of config and defaults.
        If the config value is an array, apply the default value to each element.
        If the default values dict has a value of ConfigField.REQUIRED for a key, it is required and a runtime error should be thrown if a value is not supplied from config
        """
        def _check_option(option, option_config, option_defaults):
            for key, value in option_defaults.items():
                if key not in option_config:
                    if value == ConfigField.REQUIRED:
                        raise RuntimeError("Required config value '{}' of option '{}' not supplied".format(key, option))
                    option_config[key] = value
        
        for key, value in defaults.items():
            if key not in config:
                # Then they did not pass in one of the main configs. If the default is an array or object, then we can fill it in. If is a required object, we must error
                if value == ConfigField.REQUIRED:
                    raise RuntimeError("Required config value '{}' not supplied".format(key))
                elif isinstance(value, dict):
                    config[key] = {}
                elif isinstance(value, list):
                    config[key] = [{}]
            # Config[key] is now either a dict, list of dicts, or an object that cannot be checked. 
            # If it is a list, then we need to check each element
            if isinstance(value, list):
                assert isinstance(config[key], list)
                for element in config[key]:
                    _check_option(key, element, value[0])
            elif isinstance(value, dict):
                _check_option(key, config[key], value)
            # This object does not support checking
        return config

    def get_preprocessing(self):
        """
        Takes the preprocessing dictionary and converts it to a composition of torchvision transforms
        """
        def _get_transformation(transformation_name, **kwargs):
            if transformation_name == "RandomResizedCrop":
                return T.RandomResizedCrop(**kwargs)
            elif transformation_name == "RandomHorizontalFlip":
                return T.RandomHorizontalFlip()
            elif transformation_name == "ToTensor":
                return T.ToTensor()
        
        transformations = []
        for transformation_name, transformation_kwargs in self.config["data"]["preprocessing"].items():
            if isinstance(transformation_kwargs, dict):
                transformations.append(_get_transformation(transformation_name, **transformation_kwargs))
            else:
                transformations.append(_get_transformation(transformation_name))
        return T.Compose(transformations)
    
    def __getitem__(self, key):
        return self.config[key]

# Create a simple click command line interface to load the config and start the training
@click.command()
@click.option("--config_file", default="./train_decoder_config.json", help="Path to config file")
def main(config_file):
    with open(config_file) as f:
        config = json.load(f)
    config = TrainDecoderConfig(config)
    initialize_training(config)


if __name__ == "__main__":
    main()