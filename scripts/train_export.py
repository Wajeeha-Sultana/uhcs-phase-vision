import os, glob, time, json
import numpy as np, pandas as pd, cv2
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_DIR = "./data"  # place the extracted UHCSDB archive here (see README) before running
IMG_DIR = os.path.join(DATA_DIR, "For Training", "Cropped")
META_PATH = os.path.join(DATA_DIR, "new_metadata.xlsx")
IMG_SIZE = (128,128)

def load_gray(p, size=IMG_SIZE):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def generate_pseudo_mask(g):
    blur = cv2.GaussianBlur(g, (5,5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k = np.ones((3,3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return (m>0).astype('uint8')

print("Loading data...")
all_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
all_names = [os.path.basename(p) for p in all_paths]
all_imgs = np.stack([load_gray(p) for p in all_paths]).astype('float32')/255.0
all_masks = np.stack([generate_pseudo_mask((im*255).astype('uint8')) for im in all_imgs]).astype('float32')
print("images:", all_imgs.shape)

# ---------------- U-Net ----------------
def conv_block(x, f):
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x); x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x); x = layers.BatchNormalization()(x)
    return x

def build_unet(input_shape=(128,128,1), base=12):
    inputs = layers.Input(input_shape)
    c1 = conv_block(inputs, base); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, base*2); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, base*4); p3 = layers.MaxPooling2D()(c3)
    b = conv_block(p3, base*8)
    u3 = layers.Conv2DTranspose(base*4,2,strides=2,padding='same')(b)
    u3 = layers.Concatenate()([u3,c3]); c4 = conv_block(u3, base*4)
    u2 = layers.Conv2DTranspose(base*2,2,strides=2,padding='same')(c4)
    u2 = layers.Concatenate()([u2,c2]); c5 = conv_block(u2, base*2)
    u1 = layers.Conv2DTranspose(base,2,strides=2,padding='same')(c5)
    u1 = layers.Concatenate()([u1,c1]); c6 = conv_block(u1, base)
    outputs = layers.Conv2D(1,1,activation='sigmoid')(c6)
    return models.Model(inputs, outputs, name="unet")

def dice_coef(y_true,y_pred,smooth=1.0):
    yt=tf.reshape(y_true,[-1]); yp=tf.reshape(y_pred,[-1])
    inter=tf.reduce_sum(yt*yp)
    return (2*inter+smooth)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)+smooth)
def bce_dice_loss(y_true,y_pred):
    bce=tf.keras.losses.binary_crossentropy(y_true,y_pred)
    return tf.reduce_mean(bce)+(1-dice_coef(y_true,y_pred))

X_seg = all_imgs[...,None]; y_seg = all_masks[...,None]
Xs_tr,Xs_val,ys_tr,ys_val = train_test_split(X_seg,y_seg,test_size=0.2,random_state=SEED)

unet = build_unet()
unet.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=bce_dice_loss, metrics=[dice_coef,'accuracy'])
print("Training U-Net...")
t0=time.time()
unet.fit(Xs_tr,ys_tr, validation_data=(Xs_val,ys_val), batch_size=16, epochs=10,
         callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,min_lr=1e-5)],
         verbose=2)
print("U-Net done in", time.time()-t0)
unet.save("models/unet_model.keras")

# ---------------- Classifier ----------------
meta = pd.read_excel(META_PATH)
meta['num'] = meta['path'].str.extract(r'micrograph(\d+)\.png').astype(int)
meta['cropped_path'] = meta['num'].apply(lambda n: os.path.join(IMG_DIR, f"Croppedmicrograph{n}.png"))
meta = meta[meta['cropped_path'].apply(os.path.exists)].reset_index(drop=True)

le = LabelEncoder()
meta['label_id'] = le.fit_transform(meta['primary_microconstituent'])
X_cls = np.stack([load_gray(p) for p in meta['cropped_path']]).astype('float32')/255.0
X_cls = X_cls[...,None]
y_cls = meta['label_id'].values
Xc_tr,Xc_te,yc_tr,yc_te = train_test_split(X_cls,y_cls,test_size=0.2,random_state=SEED,stratify=y_cls)
cw = compute_class_weight('balanced', classes=np.unique(yc_tr), y=yc_tr)
cwd = dict(zip(np.unique(yc_tr), cw))

def build_classifier(input_shape=(128,128,1), n_classes=6):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(16,3,activation='relu',padding='same')(inputs); x=layers.BatchNormalization()(x); x=layers.MaxPooling2D()(x)
    x = layers.Conv2D(32,3,activation='relu',padding='same')(x); x=layers.BatchNormalization()(x); x=layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x); x=layers.BatchNormalization()(x); x=layers.MaxPooling2D()(x)
    x = layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64,activation='relu')(x); x=layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes,activation='softmax')(x)
    return models.Model(inputs, outputs, name="classifier")

clf = build_classifier(n_classes=len(le.classes_))
clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training classifier...")
t0=time.time()
clf.fit(Xc_tr,yc_tr, validation_split=0.15, batch_size=16, epochs=60, class_weight=cwd,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True),
                   callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-5)],
        verbose=2)
print("Classifier done in", time.time()-t0)
clf.save("models/classifier_model.keras")

with open("models/label_classes.json","w") as f:
    json.dump(list(le.classes_), f)
with open("models/config.json","w") as f:
    json.dump({"img_size": IMG_SIZE}, f)

print("ALL DONE")
