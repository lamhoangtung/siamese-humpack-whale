TRAIN_DF = './train_up_df.csv'
SUB_Df = './sample_submission.csv'
TRAIN = './upsampling/'
# TRAIN = '/Users/lamhoangtung/whale/data/train/'
TEST = '/media/asilla/data102/hana/whale_pure/test/'
P2H = './metadata/_p2h.pickle'
P2SIZE = './metadata/_p2size.pickle'
BB_DF = "./metadata/bounding_boxes.csv"

train_batch_size = 16
train_from_scratch = False
last_weight = './model/ep240.model'
test_weight = './model/ep250.model'
img_shape = (384, 384, 3)  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
# The margin added around the bounding box to compensate for bounding box inaccuracy
crop_margin = 0.065


wrong_bb_list = ['89a1a7fae.jpg',
                 '5192f0bcf.jpg',
                 '23d2dff49.jpg',
                 '9498e6bcf.jpg',
                 'b8f420e50.jpg',
                 '6a62fab94.jpg',
                 '2d742ff03.jpg',
                 '20a08372e.jpg',
                 'd841c519b.jpg']
