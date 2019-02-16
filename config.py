TRAIN_DF = '.train_up_df.csv'
SUB_Df = './sample_submission.csv'
TRAIN = './upsampling/'
# TRAIN = '/Users/lamhoangtung/whale/data/train/'
TEST = '/media/asilla/data102/hana/whale_pure/test/'
P2H = './metadata/p2h.pickle'
P2SIZE = './metadata/p2size.pickle'
BB_DF = "./metadata/bounding_boxes.csv"

train_batch_size = 16
train_from_scratch = False
last_weight = './model/ep240.model'
test_weight = './model/ep250.model'
img_shape = (384, 384, 3)  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
# The margin added around the bounding box to compensate for bounding box inaccuracy
crop_margin = 0.05