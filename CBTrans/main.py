import time
from prepro import *
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from swinv2b import EncoderCNN
from memoryTransformer import MemoryTransformer
from data_loader import get_loader
from utils import *
from utils1 import *
from sklearn.metrics import roc_curve, roc_auc_score
from VCL import EnhancedLoss
import evaluation
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score
#from ICL_721 import EnhancedLoss
import torch.nn as nn

seed = 1234
seed_everything(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

def compute_metrics(logits, labels):
    probs = torch.softmax(logits, dim=1).cpu().detach().numpy()  # 计算预测概率
  # 计算预测概率
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    Accuracy = accuracy_score(labels, preds)
    F1 = f1_score(labels, preds, average='weighted')
    Recall = recall_score(labels, preds, average='weighted')

    return Accuracy, F1, Recall, probs, labels

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='next101_b_model_LOSS')
parser.add_argument('--model_path', type=str, default='', help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='../data_c8/vocab.pkl', help='path for vocabulary wrapper')
parser.add_argument('--image_dir', type=str, default='../data_c8/train2014', help='directory for resized images')
parser.add_argument('--image_dir_val', type=str, default='../data_c8/val2014', help='directory for resized images')
parser.add_argument('--caption_path', type=str, default='../data_c8/train_caption.json',
                    help='path for train annotation json file')
parser.add_argument('--caption_path_val', type=str, default='../data_c8/val_caption.json',
                    help='path for val annotation json file')
parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

# Model parameters
parser.add_argument('--embed_dim', type=int, default=1024, help='dimension of word embedding vectors')
parser.add_argument('--nhead', type=int, default=8, help='the number of heads in the multiheadattention models')
parser.add_argument('--num_layers', type=int, default=4,
                    help='the number of sub-encoder-layers in the transformer model')

parser.add_argument('--attention_dim', type=int, default=464, help='dimension of attention linear layers')
parser.add_argument('--decoder_dim', type=int, default=1024, help='dimension of decoder rnn')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--epochs_since_improvement', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--encoder_lr', default=0.00002, type=float,
                    help="initial learning rate")
parser.add_argument('--decoder_lr', default=0.0002, type=float,
                    help="initial learning rate")
parser.add_argument('--checkpoint', type=str, default=None, help='path for checkpoints')
parser.add_argument('--grad_clip', type=float, default=0.1)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--accumulate_best_bleu4', type=float, default=0.)
parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='fine-tune encoder')

args = parser.parse_args()


# print(args)

def main(args):
    global accumulate_bleu4, epoch, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, vocab, bleu1, bleu2, bleu3, bleu3, bleu4, rouge, cider,meteor,spice,Accuracy,F1,Recall

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # vocab = CustomUpickler(open(args.vocab_path, 'rb')).load()
    vocab_size = len(vocab)

    if args.checkpoint is None:
        decoder = MemoryTransformer(N_enc=3, N_dec=3,vocab_size = len(vocab),d_model=1024,num_classes=8)
        decoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)
        encoder = EncoderCNN()
        encoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None


    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        Accuracy = checkpoint['Accuracy']
        F1 = checkpoint['F1']
        Recall = checkpoint['Recall']
        bleu1 = checkpoint['bleu-1']
        bleu2 = checkpoint['bleu-2']
        bleu3 = checkpoint['bleu-3']
        bleu4 = checkpoint['bleu-4']
        rouge = checkpoint['rouge']
        meteor = checkpoint['meteor']
        cider = checkpoint['cider']
        spice = checkpoint['spice']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                  lr=args.encoder_lr)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = EnhancedLoss().to(device)
    # = EnhancedLoss().to(device)
    #criterion = nn.CrossEntropyLoss().to(device)

    data_transforms = {
        # data_transforms是一个字典，包含了两种数据变换（'train' 和 'valid'），分别用于训练和验证数据集。
        'train':
            transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        'valid':
            transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        # transforms.ToTensor() 和 transforms.Normalize() 用于将图像数据转换为张量并进行标准化。
    }
    # Build data loader
    train_loader = get_loader(args.image_dir, args.caption_path, vocab,
                              data_transforms['train'], args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(args.image_dir_val, args.caption_path_val, vocab,
                            data_transforms['valid'], args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    best_accumulate_bleu4 = 0.
    best_bleu1 = 0.
    best_bleu2 = 0.
    best_bleu3 = 0.
    best_bleu4 = 0.
    best_cider = 0.
    best_rouge = 0.
    best_meteor = 0.
    best_spice = 0.
    best_Accuracy = 0.
    best_F1 = 0.
    best_Recall = 0.

    for epoch in range(args.start_epoch, args.epochs):
        '''if args.epochs_since_improvement == 20:
            break'''
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        recent_Accuracy, recent_F1, recent_Recall = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion)

        is_best = recent_Accuracy > best_Accuracy



        best_Accuracy = max(recent_Accuracy, best_Accuracy)
        best_F1 = max(recent_F1, best_F1)
        best_Recall = max(recent_Recall, best_Recall)



        if recent_Accuracy > best_Accuracy:
            best_Accuracy = recent_Accuracy
        print(f'Best Accuracy score: {best_Accuracy}')
        if recent_F1 > best_F1:
            best_F1 = recent_F1
        print(f'Best F1 score: {best_F1}')
        if recent_Recall > best_Recall:
            best_Recall = recent_Recall
        print(f'Best Recall score: {best_Recall}')

        # 计算总指标Sm
        Sm = 2/7 * best_bleu4 + 1/7 * (best_rouge + best_spice + best_meteor) + 2/7 * best_cider
        print(f'Sm: {Sm}')

        if not is_best:
            args.epochs_since_improvement += 1
            print("\nEpoch since last improvement: %d\n" % (args.epochs_since_improvement,))
        else:
            args.epochs_since_improvement = 0

    save_checkpoint(args.data_name, epoch, args.epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, is_best, best_bleu1, best_bleu2, best_bleu3,
                    best_bleu4, best_rouge, best_cider,best_meteor, best_spice,best_Accuracy, best_F1, best_Recall)




def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    start = time.time()

    # Use tqdm to display progress bar
    with tqdm(total=len(train_loader)) as t:
        for i, (imgs, caps, caplens,labels) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths,logits  = decoder(imgs, caps, caplens)
            #print(scores.shape)
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = caps[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            scores = scores.data.to(device)
            targets = targets.data.to(device)
            logits = logits.to(device)
            labels = labels.to(device)
            #print("imgs: ", imgs.shape)
            #print("scores: ", scores.shape)
            #print("targets: ", targets.shape)
           # loss1 = criterion(logits, labels).to(device)
           # loss2 = criterion(scores, targets).to(device)
            #loss=loss2+loss1
            #print(loss)
            #print(loss)
            loss = criterion(imgs, scores, targets,logits,labels).to(device)




            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            if args.grad_clip is not None:
                clip_gradient(decoder_optimizer, args.grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, args.grad_clip)

            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # = accuracy(logits, labels, 1)
           # top5 = accuracy(logits, labels, 5)
            losses.update(loss.item(), sum(decode_lengths))
           # top1accs.update(top1, sum(decode_lengths))
            #top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Update tqdm progress bar
            t.set_postfix(loss=losses.avg)
            t.update()
    # Average metrics

    print('Epoch: [{0}]\t'
          'Loss {loss.avg:.4f}'.format(epoch, loss=losses,
                                          ))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: accumulate-BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    total_Accuracy = 0.0
    total_F1 = 0.0
    total_Recall = 0.0
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()
    all_preds = []  # 存储所有预测结果
    all_probs = []
    all_labels1 = []
    all_labels = []
    num_classes = 8
    per_class_correct = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}

    # Batches
    # Initialize lists to store references and hypotheses
    all_references_str = []
    all_hypotheses_str = []

    for i, (imgs, caps, caplens, labels) in enumerate(val_loader):

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        imgs = encoder(imgs)

        scores, caps_sorted, decode_lengths, logits= decoder(imgs, caps, caplens)
        scores_copy = scores.clone()
        _, preds_fen = torch.max(logits, dim=1)
        all_preds.extend(preds_fen.cpu().numpy())
        all_labels1.extend(labels.cpu().numpy())
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        #_, preds_seq = torch.max(scores, dim=2)

        targets = caps[:, 1:]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        scores = scores.data.to(device)
        targets = targets.data.to(device)
        logits = logits.to(device)
        labels = labels.to(device)

        # Compute metrics
        Accuracy, F1, Recall,probs,label_np = compute_metrics(logits, labels)
        total_Accuracy += Accuracy
        total_F1 += F1
        total_Recall += Recall
        class_preds = torch.max(logits, dim=1)[1].cpu().numpy()
        preds_flat = class_preds.flatten()
        labels_flat = label_np.flatten()

        #loss1 = criterion(logits, labels).to(device)
        #loss2 = criterion(scores, targets).to(device)
        #loss = loss2 + loss1
        loss = criterion(imgs, scores, targets, logits, labels).to(device)

        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        for c in range(num_classes):
            per_class_correct[c] += np.sum((preds_flat == c) & (labels_flat == c))
            per_class_total[c] += np.sum(labels_flat == c)

        all_probs.append(probs)
        all_labels.append(label_np)

        # Process references and hypotheses for BLEU





    cm = confusion_matrix(all_labels1, all_preds, labels=np.arange(8))  # num_classes is the total number of classes
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(8))
    # Save confusion matrix as an image
    save_path = f"./confusion_matrices_8_2e-4_1e-4/epoch_{epoch}_confusion_matrix.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Epoch {epoch + 1}')
    plt.savefig(save_path)  # Save the figure
    plt.close()  # Close the plot to free up memory


    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    n_classes = all_probs.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = roc_auc_score(all_labels == i, all_probs[:, i])

    save_dir = './8lei/2e-41e-4m10_622'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use a color map to generate a sufficient number of unique colors
    cmap = plt.get_cmap('tab10')  # 'tab10' has 10 distinct colors

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=cmap(i), lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/roc_curve_epoch_{epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(n_classes), [roc_auc[i] for i in range(n_classes)], tick_label=[f'Class {i}' for i in range(n_classes)])
    plt.xlabel('Classes')
    plt.ylabel('AUC')
    plt.title('AUC Histogram')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')
    plt.savefig(f'{save_dir}/auc_histogram_epoch_{epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    accuracy_per_class = {i: per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0 for i in per_class_total}
    bars = plt.bar(accuracy_per_class.keys(), accuracy_per_class.values())
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')
    plt.savefig(f'{save_dir}/accuracy_per_class_epoch_{epoch}.png')
    plt.close()


    Accuracy = total_Accuracy / len(val_loader)
    F1 = total_F1 / len(val_loader)
    Recall = total_Recall / len(val_loader)

    # Print evaluation results
    print('\n * LOSS - {loss.avg:.3f}\n'
          'Avg Accuracy {acc:.3f}\t'
          'Avg F1 {f1:.3f}\t'
          'Avg Recall {recall:.3f}'.format(loss=losses, acc=Accuracy, f1=F1, recall=Recall))

    return Accuracy, F1, Recall
#recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4, recent_rouge, recent_cider, recent_meteor, recent_spice,recent_Accuracy, recent_F1, recent_Recall = validate(


if __name__ == '__main__':
    main(args)

