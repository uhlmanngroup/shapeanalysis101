import time
import numpy as np
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from criterion import SupConLoss
from model import PointNet
from data_handling import get_MEF_loaders
import warnings
warnings.filterwarnings('ignore') 


def validation_loop(model, criterion, val_loader):
    model.eval()
    run_loss = 0.0
    features = []
    gt = []
    for i, data in enumerate(val_loader):
        labels, points = data
        out, embed = model(points)
        features += [embed.detach().cpu().numpy()]
        gt += [labels]
        loss = criterion(out.unsqueeze(1), labels)
        run_loss += run_loss

    features = np.concatenate(features)
    labels = np.concatenate(gt)

    skf = StratifiedKFold(n_splits=5)
    scores = []
    for train_idx, test_idx in skf.split(features, labels):
        logistic_regr = LogisticRegression(C=1, multi_class='auto', solver='lbfgs')
        logistic_regr.fit(features[train_idx], labels[train_idx])
        score = logistic_regr.score(features[test_idx], labels[test_idx])
        scores.append(score)

        preds = logistic_regr.predict(features[test_idx])
        conf_matrix = metrics.confusion_matrix(labels[test_idx], preds)
        # print("The accuracy is {0}".format(score))
        # print(conf_matrix)

    scores = np.array(scores)
    print('Avg accuracy: %.3f' % (scores.mean()))

    return run_loss / len(val_loader)

def train_loop(model, criterion, optimizer, train_loader, epoch):
    run_loss = 0.0
    for i, data in enumerate(train_loader):
        labels, points = data

        # zero parameter gradients
        optimizer.zero_grad()

        # forward
        outputs, embed = model(points)

        # loss + backward
        loss = criterion(outputs.unsqueeze(1), labels)
        loss.backward()

        # optimizer update
        optimizer.step()

        # print statistics
        run_loss += loss.item()
        #print('[%d, %d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
    avg_loss = run_loss / len(train_loader)
    print('Avg loss: %.3f' % (avg_loss))
    return avg_loss


def main(path_to_data):
    train_loader, val_loader = get_MEF_loaders(path_to_data, batch_size=100)

    # Model
    model = PointNet(embed_dim=2048, feat_dim=128, feature_transform=True)

    # criterion and optimizer
    criterion = SupConLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    for epoch in range(1, 1000):
        if epoch == 1:
            loss = validation_loop(model, criterion, val_loader)
        # train for one epoch
        time1 = time.time()
        loss = train_loop(model, criterion, optimizer, train_loader, epoch)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # validate
        if epoch % 4 == 0:
            loss = validation_loop(model, criterion, val_loader)



if __name__ == '__main__':
    path_to_data = 'data/MEF_LMNA/mef_data.npy'
    main(path_to_data)
