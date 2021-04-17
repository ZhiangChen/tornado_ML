import os
import torch
import time

def load_weights(model, weight_file_path):
    assert os.path.isfile(weight_file_path)
    model.load_state_dict(torch.load(weight_file_path))
    print("=> loaded weights '{}'".format(weight_file_path))
    return model

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    lr = AverageMeter('Lr', ':.6f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, lr, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr.update(optimizer.param_groups[0]["lr"])

        images = torch.stack(images).to(device)
        target = torch.tensor(target).to(device)

        #images = images.cuda(args.gpu, non_blocking=True)
        #target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top2.update(acc2[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, device, print_freq=100):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = torch.stack(images).to(device)
            target = torch.tensor(target).to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            #acc1 = accuracy(output, target, topk=(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))

    return top1.avg

def test(model, dataset, device):
	model.eval()
	total_nm = len(dataset)
	correct_nm = 0
	for i in range(total_nm):
		image, label = dataset[i]
		pred = model(image.unsqueeze(0).to(device))[0]
		pred = pred.to("cpu").detach().numpy().argmax()
		if pred == label:
			correct_nm += 1
	return float(correct_nm)/total_nm

def test_bin(model, dataset, device):
    model.eval()
    total_nm = len(dataset)
    correct_nm = 0
    for i in range(total_nm):
        image, label = dataset[i]
        pred = model(image.unsqueeze(0).to(device))[0]
        pred = pred.to("cpu").detach().numpy().argmax()
        pred = pred > 0
        label = label > 0
        if pred == label:
            correct_nm += 1
    return float(correct_nm)/total_nm


def infer(model, dataset, device):
    model.eval()
    total_nm = len(dataset)
    results = []
    for i in range(total_nm):
        image, label = dataset[i]
        pred = model(image.unsqueeze(0).to(device))[0]
        pred = pred.to("cpu").detach().numpy().argmax()
        results.append(pred)
    return results

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def collate_fn(batch):
    return tuple(zip(*batch))

def predict(image, model, best=False):
    model.eval()
    output = model(image)
    if best:
        pass
    return output
