d = labels_train_df[labels_train_df['image_path'] == 'images/2.jpg']
x = np.concat((d['x0'], d['x1']))
y = -np.concat((d['y0'], d['y1']))
o = np.concat((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
from sklearn.cluster import DBSCAN
dbs = DBSCAN(eps=20, min_samples=1)
t = dbs.fit_predict(o)
print(t)
plt.scatter(np.concat((d['x0'], d['x1'])), -np.concat((d['y0'], d['y1'])), c=t)
img = Image.open('images/10.jpg')
plt.imshow(img)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image  # Import PIL for image loading
import torchvision
from sklearn.cluster import DBSCAN

class PointDetectionDataset(Dataset):
  def __init__(self, images: torch.Tensor, points: list[torch.Tensor]):
    self.images = images
    self.points = points

  def __getitem__(self, index):

      return {
          'image': self.images[index],
          'points': self.points[index],
          'target': self.get_target(self.points[index])
      }
  def __len__(self):
    return len(self.images)

  @staticmethod
  def get_target(points):
    return {
      'boxes': torch.tensor([[point[0]-25, point[1]-25, point[0]+25, point[1]+25] for point in points]).float(),
      'labels': torch.tensor([1 for point in points])
    }

def create_point_dataset(df):
  images = []
  points = []
  for image_path in df['image_path'].unique():
    img = Image.open(image_path)
    d = df[df['image_path'] == image_path]
    x = np.concat((d['x0'], d['x1']))
    y = np.concat((d['y0'], d['y1']))
    o = np.concat((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    dbs = DBSCAN(eps=20, min_samples=1)
    t = dbs.fit_predict(o)
    p = []
    for i in np.unique(t):
      p.append(torch.tensor(o[t==i].mean(axis=0)))
    images.append(torchvision.transforms.functional.pil_to_tensor(img).float()/255)
    points.append(torch.stack(p))

  ds = PointDetectionDataset(images, points)
  return ds
train_ds = create_point_dataset(labels_train_df)
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import random
from copy import deepcopy

def augment_image_and_points(image, points, augmentations):
    """
    Applies a sequence of augmentations to an image (torch.Tensor) and its corresponding points using torchvision.

    Args:
        image: A torch.Tensor representing the image (e.g., from torchvision.io.read_image).  Assumed to be in CHW format.
        points: A NumPy array of shape (N, 2) representing the points (x, y).
        augmentations: A list of augmentation functions to apply.  Each function
                      should take the image and points as input and return the
                      augmented image (torch.Tensor) and augmented points.

    Returns:
        A tuple containing the augmented image (torch.Tensor) and augmented points.
    """
    augmented_image = image.clone()  # Create a copy to avoid modifying the original
    augmented_points = deepcopy(points)

    for augmentation in augmentations:
        augmented_image, augmented_points = augmentation(augmented_image, augmented_points)

    return augmented_image, augmented_points


# --- Augmentation Functions (using torchvision and torch.Tensor input) ---

def random_brightness_contrast(image, points, brightness_factor_range=(0.6, 1.6), contrast_factor_range=(0.6, 1.6)):
    """
    Applies random brightness and contrast adjustments.

    Args:
        image: A torch.Tensor (CHW format).
        points: The points as a NumPy array.
        brightness_factor_range: Tuple (min, max) for random brightness factor.
        contrast_factor_range: Tuple (min, max) for random contrast factor.

    Returns:
        The augmented image (torch.Tensor) and points.
    """
    brightness_factor = random.uniform(brightness_factor_range[0], brightness_factor_range[1])
    contrast_factor = random.uniform(contrast_factor_range[0], contrast_factor_range[1])

    augmented_image = TF.adjust_brightness(image, brightness_factor)
    augmented_image = TF.adjust_contrast(augmented_image, contrast_factor)
    return augmented_image, points

def random_gaussian_noise(image, points, mean=0, std_dev_range=(0, 0.1)):
    """
    Adds random Gaussian noise to the image.

    Args:
        image: A torch.Tensor (CHW format).
        points: The points as a NumPy array.
        mean:  Mean of the Gaussian distribution.
        std_dev_range: Tuple (min, max) for the standard deviation.

    Returns:
        The augmented image (torch.Tensor) and points.
    """
    std_dev = random.uniform(std_dev_range[0], std_dev_range[1])
    noise = torch.randn_like(image) * std_dev + mean  # Create noise tensor.
    augmented_image = torch.clamp(image + noise, 0, 1) # clamp and convert to uint8 (important)
    return augmented_image, points


def random_horizontal_flip(image, points):
    """
    Flips the image and points horizontally with a 50% probability.

    Args:
        image: A torch.Tensor (CHW format).
        points: The points as a NumPy array.

    Returns:
        The augmented image (torch.Tensor) and points.
    """
    if random.random() < 0.5:
        augmented_image = F.hflip(image)
        width = image.shape[2]  # Assuming CHW format
        augmented_points = np.array([[width - x, y] for x, y in points])
        return augmented_image, augmented_points
    else:
        return image, points


def random_vertical_flip(image, points):
    """
    Flips the image and points vertically with a 50% probability.

    Args:
        image: A torch.Tensor (CHW format).
        points: The points as a NumPy array.

    Returns:
        The augmented image (torch.Tensor) and points.
    """
    if random.random() < 0.5:
        augmented_image = F.vflip(image)
        height = image.shape[1]  # Assuming CHW format
        augmented_points = np.array([[x, height - y] for x, y in points])
        return augmented_image, augmented_points
    else:
        return image, points


def random_rotation(image, points, angle_range=(-30, 30), expand=False, center=None):
    """
    Applies a random rotation to the image and points.  Uses torchvision's rotate.

    Args:
        image: A torch.Tensor (CHW format).
        points: The points as a NumPy array.
        angle_range: Tuple (min, max) for the rotation angle in degrees.
        expand: Whether to expand the image to fit the rotation.
        center: Optional.  Center of rotation as a tuple (x, y).  If None, defaults to center of image.

    Returns:
        The augmented image (torch.Tensor) and points.
    """
    aug_points = points
    angle = random.uniform(angle_range[0], angle_range[1])

    if center is None:
      height, width = image.shape[1], image.shape[2] #CHW format
      center = (width / 2, height / 2)

    augmented_image = F.rotate(image, angle, expand=expand, center=center)

    # Apply the same transformation to the points (more complex with rotation)
    # Use the same center as the image rotation
    center_x, center_y = center
    angle_rad = np.radians(-angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    rotated_points = []
    for x, y in aug_points:
        # Translate to origin
        x -= center_x
        y -= center_y

        # Rotate
        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta

        # Translate back
        x_rot += center_x
        y_rot += center_y
        rotated_points.append([x_rot, y_rot])

    augmented_points = np.array(rotated_points)


    # Handle cases where expand=True and where points might be outside the image bounds
    if expand:
        # Calculate new image dimensions
        height, width = image.shape[1], image.shape[2]  #CHW format
        w_rot = abs(width * cos_theta) + abs(height * sin_theta)
        h_rot = abs(width * sin_theta) + abs(height * cos_theta)

        # Adjust points if they fall outside expanded image
        augmented_points[:, 0] = np.clip(augmented_points[:, 0], 0, w_rot)
        augmented_points[:, 1] = np.clip(augmented_points[:, 1], 0, h_rot)

    else:
       height, width = image.shape[1], image.shape[2] #CHW format
       augmented_points[:, 0] = np.clip(augmented_points[:, 0], 0, width)
       augmented_points[:, 1] = np.clip(augmented_points[:, 1], 0, height)


    return augmented_image, augmented_points


def random_shear(image, points, shear_factor_range=(-0.2, 0.2)):
    """
    Applies a random shear transformation to the image and points.
    This *approximates* shear using an affine transform.  It's not a perfect shear
    because the transformation is implemented with `torchvision.transforms.functional.affine` which
    uses affine transformations.  Real shear is not an affine transform.
    For more accurate shear, you might need to use a different library or implementation.

    Args:
        image: A torch.Tensor (CHW format).
        points: The points as a NumPy array.
        shear_factor_range: Tuple (min, max) for the shear factor.

    Returns:
        The augmented image (torch.Tensor) and points.
    """
    shear_factor = random.uniform(shear_factor_range[0], shear_factor_range[1])

    # torchvision affine transform expects a matrix.  We'll create a shear matrix.
    # Note: This is an affine transformation, which *approximates* shear.
    shear_matrix = [1, shear_factor]  # [a, b, c, d, e, f] for shear in x direction

    height, width = image.shape[1], image.shape[2]
    center_x, center_y = width / 2, height / 2

    # torchvision affine transforms:
    augmented_image = F.affine(
        image,
        angle=0, # shear doesn't use rotation
        translate=(0, 0),  # No translation
        scale=1.0,          # No scaling
        shear=shear_matrix,
        center=(center_x, center_y),  # Shear around the center
    )

    # Apply the same transformation to the points.  Needs the inverse matrix for correct transformation
    # Calculate the inverse shear matrix.  This is needed to transform the keypoints properly.
    shear_matrix_np = np.array([[1, shear_factor, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    shear_matrix_inv_np = np.linalg.inv(shear_matrix_np)
    shear_matrix_inv = [shear_matrix_inv_np[0,0], shear_matrix_inv_np[0,1], shear_matrix_inv_np[0,2],
                       shear_matrix_inv_np[1,0], shear_matrix_inv_np[1,1], shear_matrix_inv_np[1,2]]

    rotated_points = []
    for x, y in points:
        # Convert to homogeneous coordinates
        homogeneous_point = np.array([x, y, 1])

        # Apply the inverse shear transform
        transformed_point = np.dot(shear_matrix_inv_np, homogeneous_point)
        x_trans, y_trans = transformed_point[0], transformed_point[1]

        rotated_points.append([x_trans, y_trans])

    augmented_points = np.array(rotated_points)
    # Clip to image boundaries
    augmented_points[:, 0] = np.clip(augmented_points[:, 0], 0, width)
    augmented_points[:, 1] = np.clip(augmented_points[:, 1], 0, height)
    return augmented_image, augmented_points


def random_crop(image, points, crop_factor=(0.6, 1.0)):
    """
    Randomly crops the image and adjusts the points accordingly.

    Args:
        image: A torch.Tensor (CHW format).
        points: The points as a NumPy array.
        crop_factor: Tuple (min, max) for the crop factor (relative to image size).

    Returns:
        The augmented image (torch.Tensor) and points.
    """
    height, width = image.shape[1], image.shape[2]
    crop_scale = random.uniform(crop_factor[0], crop_factor[1])

    # Determine crop dimensions
    crop_width = int(width * crop_scale)
    crop_height = int(height * crop_scale)

    # Determine the starting points of the crop
    x1 = random.randint(0, width - crop_width)
    y1 = random.randint(0, height - crop_height)
    x2 = x1 + crop_width
    y2 = y1 + crop_height

    # Crop the image using F.crop (torchvision.transforms.functional.crop)
    augmented_image = F.crop(image, y1, x1, crop_height, crop_width)

    # Adjust the points to the new coordinate system
    augmented_points = points - np.array([x1, y1])

    # Filter points outside the cropped image
    valid_indices = (augmented_points[:, 0] >= 0) & (augmented_points[:, 0] < crop_width) & \
                    (augmented_points[:, 1] >= 0) & (augmented_points[:, 1] < crop_height)
    augmented_points = augmented_points[valid_indices]

    return augmented_image, augmented_points


# --- Example Usage ---
if __name__ == '__main__':
    image = train_ds[0]['image']
    points = train_ds[0]['points']
    # 2. Define the augmentations
    augmentations = [
        random_brightness_contrast,
        random_gaussian_noise,
        random_horizontal_flip,
        random_vertical_flip,
        random_rotation,
        # random_shear,
        # random_crop,
    ]

    # 3. Apply augmentations
    num_augmentations = 4
    for i in range(num_augmentations):
        # Select a random subset of augmentations
        selected_augmentations = random.sample(augmentations, random.randint(1, len(augmentations)))
        augmented_image, augmented_points = augment_image_and_points(image, points, augmentations)


    print("Done.")

from torchvision.models.detection import ssd300_vgg16
def create_point_detection_model():
    """
    Creates an SSD model for point detection.

    Args:
        num_classes (int): Number of classes (including background).  Typically 2 (background, point).
        pretrained (bool): Whether to use a pretrained backbone.
        trainable_backbone_layers (int, optional):  Number of trainable layers in the backbone.
                                                    If None, all layers are trainable.
                                                    Defaults to None.

    Returns:
        torch.nn.Module: The SSD model.
    """
    # Load the SSD300 model with VGG16 backbone
    model = ssd300_vgg16(weights=False, weights_backbone=False, num_classes=2)

    # model.head.classification_head = torch.nn.Conv2d(in_channels, 2, kernel_size=3, padding=1) #changed to match num_classes

    # in_channels = model.head.bbox_head.conv.in_channels
    # num_anchors = len(model.head.bbox_head.anchors_generator.aspect_ratios) #3, 3, 3, 3, 3, 3
    # model.head.bbox_head.conv = torch.nn.Conv2d(in_channels, num_anchors * 2, kernel_size=3, padding=1) #change output to 2 coords instead of 4

    return model

valid_ds = create_point_dataset(labels_validation_df)

augmentations = [
    random_brightness_contrast,
    random_gaussian_noise,
    random_horizontal_flip,
    random_vertical_flip,
    random_rotation,
]


model = create_point_detection_model()


model.train()
device = 'cuda'
model = model.to(device)
def collate_fn(batch):
    """
    Custom collate function to handle variable sized images and targets.
    """
    # print(type(batch))
    images = []
    points = []
    target = []
    for i in batch:
      augmented_image, augmented_points = augment_image_and_points(i['image'].to(device), i['points'], augmentations)

      images.append(augmented_image)
      points.append(augmented_points)
      target.append(PointDetectionDataset.get_target(augmented_points))
    return {
        'image': torch.stack(images),
        'points': points,
        'target': target
    }

def collate_no_aug(batch):
    return {
        'image': torch.stack([i['image'] for i in batch]),
        'points': [i['points'] for i in batch],
        'target': [i['target'] for i in batch]
    }
    return tuple(zip(*batch))

dataloader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_ds, batch_size=16, shuffle=False, collate_fn=collate_no_aug)

optim = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.00005)

for epoch in range(150):
  loss_acum = 0
  loss_valid_acum = 0
  for batch in tqdm(dataloader):
    images = batch['image'].to(device)
    targets = [{key: value.to(device) for key, value in t.items()}for t in batch['target']]

    optim.zero_grad()
    out = model(images, targets)
    loss = sum(out.values())
    loss.backward()
    optim.step()
    loss_acum += loss.item()*images.shape[0]
  with torch.no_grad():
    for batch in tqdm(valid_dataloader):
      images = batch['image'].to(device)
      targets = [{key: value.to(device) for key, value in t.items()}for t in batch['target']]

      out = model(images, targets)
      loss = sum(out.values())
      loss_valid_acum += loss.item()*images.shape[0]

  print(epoch, loss_acum/len(train_ds), loss_valid_acum/len(valid_ds))

model.eval()
m_out = model(train_ds[0]['image'].unsqueeze(0).to(device))[0]
print(m_out)

torch.abs(m_out['boxes'][:, 0] - m_out['boxes'][:, 2]).min()
bx = valid_ds[0]['target']['boxes']
sample = valid_ds[7]
m_out = model(sample['image'].unsqueeze(0).to(device))[0]
bx = m_out['boxes'][m_out['scores']>0.5]
plt.imshow(sample['image'].permute(1, 2, 0))
plt.scatter([p[0] for p in sample['points']], [p[1] for p in sample['points']], s=20)
# plt.scatter((bx[:, 0]+bx[:, 2]).detach().cpu()/2, (bx[:, 1]+bx[:, 3]).detach().cpu()/2, s=10)
plt.scatter((bx[:, 0]).detach().cpu(), (bx[:, 1]).detach().cpu(), s=10)
plt.scatter((bx[:, 2]).detach().cpu(), (bx[:, 3]).detach().cpu(), s=10)

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageDraw
import math
def process_segment(image_tensor, x1, y1, x2, y2, segment_width=16, output_size=(16, 128)):
    """
    Rotates, crops, and resizes an image to focus on a line segment.

    Args:
        image_path (str): Path to the input image.
        x1, y1, x2, y2 (float): Coordinates of the line segment endpoints.
        segment_width (int): Desired width (in pixels) of the segment in the cropped image.
        output_size (tuple): Desired output size (width, height) of the final image.

    Returns:
        torch.Tensor: A PyTorch tensor representing the processed image, or None if there's an error.
    """
    dx = x2 - x1
    dy = y2 - y1
    mx = (x1+x2)/2
    my = (y1+y2)/2
    angle_rad = math.atan2(dy, dx)  # Angle in radians
    angle_deg = math.degrees(angle_rad)  # Angle in degrees
    segment_length = math.sqrt(dx**2 + dy**2)

    # 2. Get image dimensions
    _, height, width = image_tensor.shape

    # 3. Rotate the image
    rotated_tensor = TF.rotate(image_tensor, angle_deg, expand=False, center=[(y1+y2)/2, (x1+x2)/2][::-1]) # Rotate counter-clockwise

    # return TF.crop(rotated_tensor, int(my)-output_size[0]//2, int(mx)-int(output_size[1]/2), output_size[0], int(output_size[1]))
    cropped =  TF.crop(rotated_tensor, int(my)-segment_width//2, int(mx)-int(segment_length/2), segment_width, int(segment_length))
    return TF.resize(cropped, output_size)


    # # Create a bounding box with width and height based on segment width and length
    # # Use max to avoid out-of-bounds calculations.
    # bounding_box_width = int(max(segment_length, segment_width)) + 20  # Add a little padding (e.g. 10px on each side)
    # bounding_box_height = int(segment_width * 2)  # Segment width doubled

    # # Calculate the center of the segment in the rotated image
    # center_x_rotated = width_rotated / 2
    # center_y_rotated = height_rotated / 2

    # # Calculate the crop box
    # crop_x = int(center_x_rotated - bounding_box_width / 2)
    # crop_y = int(center_y_rotated - bounding_box_height / 2)

    # # 5. Crop the rotated image
    # cropped_tensor = TF.crop(rotated_tensor, crop_y, crop_x, bounding_box_height, bounding_box_width)

    # # 6. Resize the cropped image
    # resized_tensor = TF.resize(cropped_tensor, output_size)

    # return resized_tensor
segments = np.array(labels_train_df[labels_train_df['image_path']=='images/10.jpg'])
img = Image.open('images/10.jpg')
img = torchvision.transforms.functional.pil_to_tensor(img).float()/255
df = labels_train_df
[TYPE_INDEX_MAP[x] +1 for x in df[df['image_path']=='images/10.jpg']['type']]
class EdgeDataset(Dataset):
  def __init__(self, df):
    self.files = df['image_path'].unique()
    self.images = []
    self.edges = []
    self.labels = []
    for i in self.files:
      img = Image.open(i)
      img = torchvision.transforms.functional.pil_to_tensor(img).float()/255
      self.images.append(img)
      self.edges.append(torch.tensor(df[df['image_path']==i][['x0', 'y0', 'x1', 'y1']].to_numpy()))
      self.labels.append(torch.tensor([TYPE_INDEX_MAP[x] +1 for x in df[df['image_path']==i]['type']]))
  def __getitem__(self, index):
      return {
          'image': self.images[index],
          'edges': self.edges[index],
          'labels': self.labels[index]
      }
  def __len__(self):
    return len(self.files)

  @staticmethod
  def get_images(item, std=0.0, bg=0.0):
    o = []
    for i in item['edges']:
      pos = i + torch.randn((4,)) * std

      o.append(process_segment(item['image'], pos[0], pos[1], pos[2], pos[3]))
    n_bg = int(len(item['edges'])*bg)
    for i in range(n_bg):
      i1 = random.randint(0, len(item['edges'])-1)
      i2 = random.randint(0, len(item['edges'])-1)
      while i1 == i2:
        i2 = random.randint(0, len(item['edges'])-1)
      b1 = random.randint(0, 1)
      b2 = random.randint(0, 1)


      e = torch.tensor([item['edges'][i1, b1*2], item['edges'][i1, b1*2+1], item['edges'][i2, b2*2], item['edges'][i2, b2*2+1]])
      e2 = torch.tensor([item['edges'][i2, b2*2], item['edges'][i2, b2*2+1], item['edges'][i1, b1*2], item['edges'][i1, b1*2+1]])

      while min(torch.sum((item['edges']-e)**2, axis=1).min(), torch.sum((item['edges']-e2)**2, axis=1).min()) < 500 or ((e[0] - e[2])**2 + (e[1] - e[3])**2) < 50:
        i1 = random.randint(0, len(item['edges'])-1)
        i2 = random.randint(0, len(item['edges'])-1)
        while i1 == i2:
          i2 = random.randint(0, len(item['edges'])-1)
        b1 = random.randint(0, 1)
        b2 = random.randint(0, 1)


        e = torch.tensor([item['edges'][i1, b1*2], item['edges'][i1, b1*2+1], item['edges'][i2, b2*2], item['edges'][i2, b2*2+1]])
        e2 = torch.tensor([item['edges'][i2, b2*2], item['edges'][i2, b2*2+1], item['edges'][i1, b1*2], item['edges'][i1, b1*2+1]])

      o.append(process_segment(item['image'], *e))


    return torch.stack(o), torch.cat((item['labels'], torch.tensor([0 for i in range(n_bg)])))
train_edge_ds = EdgeDataset(labels_train_df)
valid_edge_ds = EdgeDataset(labels_validation_df)
import torch.nn as nn

class EdgeClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(EdgeClassifier, self).__init__()

        # Convolutional layers (consider increasing filters or adding more layers)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Output: (32, 64, 128)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (32, 32, 64)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Output: (64, 32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 16, 32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Output: (128, 16, 32)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (128, 8, 16)



        # Fully connected layers
        # Calculate input features for the first FC layer.  This is crucial!
        self.fc_input_size = 128 * 2 * 16  # Calculated based on the output of the convolutional layers

        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout = nn.Dropout(0.4) # Dropout for regularization
        self.fc2 = nn.Linear(256, num_classes) # Output layer


    def forward(self, x):
        # Convolutional and pooling layers
        # x = nn.functional.relu(self.bn1(self.conv1(x)))
        # x = self.pool1(x)
        # x = nn.functional.relu(self.bn2(self.conv2(x)))
        # x = self.pool2(x)
        # x = nn.functional.relu(self.bn3(self.conv3(x)))
        # x = self.pool3(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool3(x)


        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten: batch_size x (features)


        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation on the final layer for classification (use CrossEntropyLoss)
        return x
edge_model = EdgeClassifier().to(device)
class Edataset(Dataset):
  def __init__(self, edgeDataset: EdgeDataset, transform = None):
    self.transform = transform if transform is not None else lambda x: x
    self.images = []
    self.labels = []
    for _ in range(5):
      for i in edgeDataset:
        x, y = edgeDataset.get_images(i, 3, 0.25)
        self.images.append(x)
        self.labels.append(y)
    self.images = torch.cat(self.images)
    self.labels = torch.cat(self.labels)
  def __getitem__(self, index):
    return self.transform(self.images[index]), self.labels[index]
  def __len__(self):
    return self.images.shape[0]
train_eds = Edataset(train_edge_ds, transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.03, hue=0.005)]))
valid_eds = Edataset(valid_edge_ds)

train_edge_dataloader = DataLoader(train_eds, 32, shuffle=True)
valid_edge_dataloader = DataLoader(valid_eds, 16, shuffle=False)

edge_model.train()
optim_edge = torch.optim.Adam(edge_model.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(40):
  loss_acum = 0
  loss_valid_acum = 0
  samples_acum = 0
  samples_valid_acum = 0
  correct_valid_acum = 0
  for images, labels in tqdm(train_edge_dataloader):
    # print(labels)
    # for image, label in zip(images, labels):
    #   plt.imshow(image.cpu().permute(1, 2, 0))
    #   plt.title(label)
    #   plt.show()
    # break
    images = images.to(device)
    optim_edge.zero_grad()
    out = edge_model(images)
    loss = criterion(out, labels.to(device).long())
    loss.backward()
    optim_edge.step()
    loss_acum += loss.item()*images.shape[0]
    samples_acum += images.shape[0]
  model.eval()
  with torch.no_grad():
    for images, labels in tqdm(valid_edge_dataloader):
      images = images.to(device)

      out = edge_model(images)
      loss = criterion(out, labels.to(device).long())
      loss_valid_acum += loss.item()*images.shape[0]
      samples_valid_acum += images.shape[0]
      correct_valid_acum += (out.argmax(axis=1) == labels.to(device).long()).sum().item()
  model.train()
  print(epoch, loss_acum/samples_acum, loss_valid_acum/samples_valid_acum, correct_valid_acum/samples_valid_acum)
def your_solution(images_path):
    predictions = {}
    type_map_r = {value: key for key, value in TYPE_INDEX_MAP.items()}

    for img_path in tqdm(images_path):
        detections = []


        img = Image.open(img_path)
        img = torchvision.transforms.functional.pil_to_tensor(img).float()/255
        m_out = model(img.to(device).unsqueeze(0))[0]
        bx = m_out['boxes'][m_out['scores']>0.2]
        px = (bx[:, 0] + bx[:, 2])/2
        py = (bx[:, 1] + bx[:, 3])/2
        neigh = [[] for _ in range(px.shape[0])]
        for p1 in range(px.shape[0]):
          for p2 in range(p1+1, px.shape[0]):
            d = math.sqrt((px[p1] - px[p2]) ** 2 + (py[p1] - py[p2])**2)
            if d < 20 or d > 180:
              continue
            pr = process_segment(img, px[p1], py[p1], px[p2], py[p2])
            e = edge_model(pr.to(device).unsqueeze(0))[0].argmax()
            if e!=0:
              neigh[p1].append((p2, type_map_r[e.item()-1]))
              neigh[p2].append((p1, type_map_r[e.item()-1]))

              if px[p1] < px[p2]:
                detections.append({'x0': px[p1].item(), 'y0': py[p1].item(), 'x1': px[p2].item(), 'y1': py[p2].item(), 'type': type_map_r[e.item()-1]})
              else:
                detections.append({'x0': px[p2].item(), 'y0': py[p2].item(), 'x1': px[p1].item(), 'y1': py[p1].item(), 'type': type_map_r[e.item()-1]})
        # print(neigh)
        odw = [0 for i in range(px.shape[0])]
        start = py.argmin()

        def dfs(v):
          odw[v] = 1
          res = []
          for i, t in neigh[v]:
            if odw[i]:
              continue
            res.append((t, dfs(i)))
          return res

        # return
        tree = dfs(start)
        predictions[img_path] = {
            'detections': detections,
            'tree': ('root', tree)
        }


    return predictions