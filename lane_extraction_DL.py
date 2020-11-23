from torchvision import transforms
from PIL import Image
#from utils import *
import torch.nn as nn
import cv2
from torchvision.utils import *
import matplotlib.pyplot as plt
from utils import *

device = torch.device('cuda')

class UNet_ConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_ConvLSTM, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.convlstm = ConvLSTM(input_size=(8,16),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            data.append(x5.unsqueeze(0))
        data = torch.cat(data, dim=0)
        lstm, _ = self.convlstm(data)
        test = lstm[0][ -1,:, :, :, :]
        x = self.up1(test, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, test


class process():
    def __init__(self):
        self.tranforms = transforms.Compose([transforms.ToTensor()])


def show(img_0, img_1):
    fig = plt.figure()

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(img_0)
    plt.title("Original")

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(img_1)
    plt.title("MeanShift")
    plt.show()

def findSignificantContour(edgeImg):
    contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
        # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
# From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour


def removesmall(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    min_size = 1500

    img2 = np.zeros((img.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img2

if __name__ == '__main__':


    img = cv2.imread('lane4.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    op_tranforms = transforms.Compose([transforms.ToTensor()])
    img = cv2.resize(img, (256, 128))
    img = op_tranforms(img).detach().numpy()
    img = np.array([img])

    data = torch.unsqueeze(torch.tensor(img), dim=0).to(device)
    model = UNet_ConvLSTM(3, 2).to(device)

    pretrained_dict = torch.load('./best.pth')
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)



    output, feature = model(data)
    pred = output.max(1, keepdim=True)[1]
    img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
    cv_img = img.astype(np.uint8)

    img = cv2.resize(cv_img, (1280, 720))
    dmy = img.copy()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 30, maxLineGap=200)

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)  # <-- Calculating the slope.
            if math.fabs(slope) < 0.3:  # <-- Only consider extreme slope
                continue
            if slope <= 0:  # <-- If the slope is negative, left group.
                print("left")
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
                cv2.line(dmy, (x1, y1), (x2, y2), (0, 0, 255), 15)
            else:  # <-- Otherwise, right group.
                print("right")
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
                cv2.line(dmy, (x1, y1), (x2, y2), (0, 255,0), 15)

    hsv = cv2.cvtColor(dmy, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10, 255, 255])

    lower_green = (60, 255, 255)
    higher_green = (60, 255, 255)

    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    mask2 = cv2.inRange(hsv, lower_green, higher_green)
    show(dmy, mask1)
    show(dmy, mask2)