from utils import parse_args, create_logger, DEVICE, fix_random_seed

from dataset import GraphDataset
from dataloader import TorchDataloader
from loss import MRGSRecLoss
from model import MRGSRecModel
from optimizer import BasicOptimizer

import copy
import json
import torch

logger = create_logger(name=__name__)
seed_val = 42


def train(dataloader, model, optimizer, loss_function, epoch_cnt=None, step_cnt=None, best_metric=None):
    step_num = 0
    epoch_num = 0
    current_metric = 0

    epochs_threshold = 40

    best_epoch = 0
    best_checkpoint = None

    logger.debug('Start training...')

    while (epoch_cnt is None or epoch_num < epoch_cnt) and (step_cnt is None or step_num < step_cnt):
        if best_epoch + epochs_threshold < epoch_num:
            logger.debug('There is no progress during {} epochs. Finish training'.format(epochs_threshold))
            break

        logger.debug(f'Start epoch {epoch_num}')
        for step, batch in enumerate(dataloader):
            batch_ = copy.deepcopy(batch)

            model.train()

            for key, values in batch_.items():
                batch_[key] = batch_[key].to(DEVICE)

            batch_.update(model(batch_))
            loss = loss_function(batch_)

            optimizer.step(loss)
            step_num += 1

            if best_metric is None:
                # Take the last model
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = epoch_num
            elif best_checkpoint is None or best_metric in batch_ and current_metric <= batch_[best_metric]:
                # If it is the first checkpoint, or it is the best checkpoint
                current_metric = batch_[best_metric]
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = epoch_num

        epoch_num += 1
    logger.debug('Training procedure has been finished!')
    return best_checkpoint


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))
    logger.debug('Current DEVICE: {}'.format(DEVICE))

    dataset = GraphDataset.create_from_config(config['dataset'])

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

    train_dataloader = TorchDataloader.create_from_config(
        config['dataloader']['train'],
        dataset=train_sampler,
        **dataset.meta
    )

    # validation_dataloader = TorchDataloader.create_from_config(
    #     config['dataloader']['validation'],
    #     dataset=validation_sampler,
    #     **dataset.meta
    # )
    #
    # eval_dataloader = TorchDataloader.create_from_config(
    #     config['dataloader']['validation'],
    #     dataset=test_sampler,
    #     **dataset.meta
    # )

    model = MRGSRecModel.create_from_config(config['model'], **dataset.meta).to(DEVICE)
    loss_function = MRGSRecLoss.create_from_config(config['loss'])
    optimizer = BasicOptimizer.create_from_config(config['optimizer'], model=model)

    logger.debug('Everything is ready for training process!')

    # Train process
    _ = train(
        dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch_cnt=config.get('train_epochs_num'),
        step_cnt=config.get('train_steps_num'),
        best_metric=config.get('best_metric')
    )

    logger.debug('Saving model...')
    checkpoint_path = '../checkpoints/{}_final_state.pth'.format(config['experiment_name'])
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug('Saved model as {}'.format(checkpoint_path))


if __name__ == '__main__':
    main()
