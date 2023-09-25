import time

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor, BasePostprocessor
from openood.utils import setup_logger
import torch
from tqdm import tqdm

class TestOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        net = get_network(self.config.network)

        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)
        if self.config.evaluator.ood_scheme == 'fsood':
            acc_metrics = evaluator.eval_acc(
                net,
                id_loader_dict['test'],
                postprocessor,
                fsood=True,
                csid_data_loaders=ood_loader_dict['csid'])
        else:
            acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
                                             postprocessor)
        print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
              flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection methods
        timer = time.time()
        if self.config.evaluator.ood_scheme == 'fsood':
            evaluator.eval_ood(net,
                               id_loader_dict,
                               ood_loader_dict,
                               postprocessor,
                               fsood=True)
        else:
            evaluator.eval_ood(net, id_loader_dict, ood_loader_dict,
                               postprocessor)
        print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)

class TestFLOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self, net):
        # generate output directory and save the full config file
        # setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        # ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        # net = get_network(self.config.network)

        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        MSPprocessor = BasePostprocessor(self.config)
        # setup for distance-based methods
        # postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        net.eval()

        # threshold based on train datasets
        data_loader = id_loader_dict['train']
        postprocessor.setup(net, id_loader_dict, None)
        result = []
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()
                pred, score = postprocessor.postprocess(net, data)
                result.append(score)
        score_all = torch.cat(result)
        k = int(len(score_all)*0.05)
        threshold_value, _ = torch.kthvalue(score_all, k)

        # mean = torch.mean(score_all)
        # std = torch.std(score_all)
        # threshold_value = mean - 3*std

        loss_avg = 0.0
        correct = 0
        data_loader = id_loader_dict['test']
        result = []
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()
                pred, score = postprocessor.postprocess(net, data)
                _, conf = MSPprocessor.postprocess(net, data)
                conf = conf.cpu()
                conf = torch.where(score.cpu()>threshold_value.cpu(), conf, 0)
                result.append(dict({'pred':pred, 'conf':conf, 'target':target}))

        # print('\n', flush=True)
        # print(u'\u2500' * 70, flush=True)

        # # start calculating accuracy
        # print('\nStart evaluation...', flush=True)
        # if self.config.evaluator.ood_scheme == 'fsood':
        #     acc_metrics = evaluator.eval_acc(
        #         net,
        #         id_loader_dict['test'],
        #         postprocessor,
        #         fsood=True,
        #         csid_data_loaders=ood_loader_dict['csid'])
        # else:
        #     acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
        #                                      postprocessor)
        # print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
        #       flush=True)
        # print(u'\u2500' * 70, flush=True)

        # # start evaluating ood detection methods
        # timer = time.time()
        # if self.config.evaluator.ood_scheme == 'fsood':
        #     evaluator.eval_ood(net,
        #                        id_loader_dict,
        #                        ood_loader_dict,
        #                        postprocessor,
        #                        fsood=True)
        # else:
        #     evaluator.eval_ood(net, id_loader_dict, ood_loader_dict,
        #                        postprocessor)
        # print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)
        return result
