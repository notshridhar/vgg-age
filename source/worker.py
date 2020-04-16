import time
import shutil
import torch

# global variables
best_acc1 = 0

# Average Value Computer Class
class AverageMeter(object):
    def __init__(self):
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


def _train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch train function

    parameters -------------------------
    - train_loader  -   train data generator object
    - model         -   torch model object
    - criterion     -   loss function object
    - optimizer     -   optimizer object
    - epoch         -   epoch number to train

    returns ----------------------------
    - None
    """

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    cuda_exists = torch.cuda.is_available()
    len_train = len(train_loader)

    # switch to train mode
    model.train()
    print("")
    print("EPOCH : {}".format(epoch))

    for i, (input, target) in enumerate(train_loader):

        if cuda_exists:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = _accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 20 bars to display progress
        bar = (20 * (i + 1)) // len_train

        print(
            "\r"
            "(" + str(i + 1) + "/" + str(len_train) + ")"
            "[" + "=" * bar + "_" * (20 - bar) + "]       "
            "Loss: {loss.val:.4f} ({loss.avg:.4f})        "
            "Acc@1: {top1.val:.3f} ({top1.avg:.3f})       "
            "Acc@5: {top5.val:.3f} ({top5.avg:.3f})".format(
                loss=losses, top1=top1, top5=top5,
            ),
            end="",
        )

    print("")


def _validate(valid_loader, model, criterion):
    """
    Validation function

    parameters -------------------------
    - valid_loader  -   validation data generator object
    - model         -   torch model object
    - criterion     -   loss function object

    returns ----------------------------
    - top1.avg      -   top 1 average accuracy
    """

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    cuda_exists = torch.cuda.is_available()
    len_valid = len(valid_loader)

    # switch to evaluate mode
    model.eval()
    print("VALIDATION :")

    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):

            if cuda_exists:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = _accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # 20 bars to display progress
            bar = (20 * (i + 1)) // len_valid

            print(
                "\r"
                "(" + str(i + 1) + "/" + str(len_valid) + ")"
                "[" + "=" * bar + "_" * (20 - bar) + "]       "
                "Loss: {loss.val:.4f} ({loss.avg:.4f})        "
                "Acc@1: {top1.val:.3f} ({top1.avg:.3f})       "
                "Acc@5: {top5.val:.3f} ({top5.avg:.3f})".format(
                    loss=losses, top1=top1, top5=top5,
                ),
                end="",
            )

        print("")

    return top1.avg


def _accuracy(output, target, topk=(1,)):
    """
    Compute the accuracy over the k top predictions

    parameters -------------------------
    - output        -   model output tensor
    - target        -   actual label tensor
    - topk          -   top k accuracy values to return

    returns ----------------------------
    - res           -   list of k top accuracies
    """

    num_classes = 1
    for dim in output.shape[1:]:
        num_classes *= dim

    with torch.no_grad():
        maxk = max(topk)
        maxk = min(maxk, num_classes)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k < num_classes:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                res.append([0, 0])

        return res


def train(
    model,
    loaders,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    epochs=10,
    checkpoint=None,
):
    """
    The main worker function used to train network

    parameters -------------------------
    - model         -   torch nn module
    - loaders       -   tuple of train and validation DataLoader
    - lr            -   learning rate of model
    - momentum      -   weighted average coefficient (alpha)
    - weight_decay  -   decay of weights coefficient (eta)
    - epochs        -   number of iterations to train
    - checkpoint    -   checkpoint dict

    returns ----------------------------
    - None
    """

    global best_acc1

    # create model
    print("=> training", model.name)

    # unpack loaders
    train_loader, valid_loader = loaders

    # find device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("=> found cuda compatible gpu")
    else:
        device = torch.device("cpu")
        print("=> no cuda devices found, using cpu for training")

    # device switches and optimization
    torch.backends.cudnn.benchmark = True

    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum, weight_decay=weight_decay,
    )

    # resume from a checkpoint
    if checkpoint:
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> loaded checkpoint", end=" ")
        print("with epoch = %d" % start_epoch, end=" ")
        print("and accuracy = %.2f" % best_acc1)
    else:
        start_epoch = 0

    crtm = time.ctime().split()[1:-1]
    print("=> checkpoints will be saved as checkpoint.pth")
    print("=> training started at %s-%s %s" % (crtm[0], crtm[1], crtm[2]))

    # training
    for epoch in range(start_epoch, epochs):

        # adjust learning rate
        lr_adj = lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_adj

        # train for one epoch
        _train(
            train_loader, model, criterion, optimizer, epoch,
        )

        # remember best accuracy
        acc1 = _validate(valid_loader, model, criterion)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save checkpoint
        save_dict = {
            "epoch": epoch + 1,
            "arch": model.name,
            "best_acc1": best_acc1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(save_dict, "checkpoint.pth")


def confusion_matrix(model, valid_loader):
    """
    Obtain confusion matrix from prediction
    and actual labels
    """

    len_valid = len(valid_loader)
    cuda_exists = True if torch.cuda.is_available() else False

    # confusion matrix of ncls * ncls
    ncls = model.num_classes
    conf_matrix = torch.zeros(ncls, ncls)

    # switch to evaluate mode
    model.eval()
    print("VALIDATION :")

    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):

            # compute output
            if cuda_exists:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = model(input)
            _, preds = torch.max(output, 1)

            for t, p in zip(target.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

            # 20 bars to display progress
            bar = (20 * (i + 1)) // len_valid

            print(
                "\r"
                "(" + str(i + 1) + "/" + str(len_valid) + ")"
                "[" + "=" * bar + "_" * (20 - bar) + "]",
                end="",
            )
        print("")

    # horiz normalization to get percentage
    norm_conf = []
    for row in conf_matrix:
        factor = float(row.sum())
        normed = [float(i) / factor for i in row]
        norm_conf.append(normed)

    return norm_conf
