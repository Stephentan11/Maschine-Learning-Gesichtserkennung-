import os
import time
import uuid
import cv2
import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model



#Dieser Code wird von Bibliothek os, cv2, uuid und time importiert, dessen Aufnahme von Bildern mithilfe einer Webcam ermöglicht.
#Jedes aufgenommene Bild wird in angegebenen Pfad (C:\Face Recognition\data\image) gespeichert. Erstmal werden 30 Bilder aufgenommen und bei erfolgreicher Aufnahme wird noch zweimal ausgeführt. Insgesamt ergibt sich 90 Bildern.
"""
images_path = os.path.join('data', 'image')
number_images = 30

capture = cv2.VideoCapture(0)
for imagenumber in range (number_images):
    print("collecting images {}",format(imagenumber))
    ret, frame = capture.read()
    imagename = os.path.join(images_path,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imagename, frame)
    cv2.imshow("frame", frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
capture.release()
cv2.destroyAllWindows()
"""

#Für diesen Schritt wird ein Werkzeug aus labelme genommen, womit alle Bildern manuell markiert werden, um in JSON-Datei zu transformieren. Markiert werden die Bilder mit Gesicht.
"""!labelme"""

#Bild in die TF-Datenpipeline laden
images = tf.data.Dataset.list_files('data\\image\\*.jpg')

print("Dataset created successfully")

#Es wird die erste Bilddatei abgerufen
first_image_file = next(iter(images))
print(first_image_file) #see that my images are loaded



def load_image(x):
    byte_image = tf.io.read_file(x)
    image = tf.io.decode_jpeg(byte_image)
    return image
images = images.map(load_image) #map-Function wird zum mapdatasets transformiert

print(type(images))

#Rohbilder mit Matplotlib anzeigen
imagegenerator =iter(images.batch(4).as_numpy_iterator())
plot_images = next(imagegenerator)
figure, ax = plt.subplots(ncols=4, figsize=(20,20))
for index, image in enumerate(plot_images):
    ax[index].imshow(image)
plt.show()

# 60 Fotos zum Train Order, 15 Fotos zum Test Ordner und 15 Fotos zum validate Ordner(Manually copy paste von data Ordner(images))

# Aus dem Ordner (1. label) werden in den 3 weiteren Ordern (test, train, validate) geschoben, wobei die Bilder mit deren Namen angepasst werden.
'''
for folder in['train', 'test', 'validate']:
    for file in os.listdir(os.path.join('data', folder, 'image')):

        filename = file.split('.')[0]+'.json'
        exist_filepath = os.path.join('data','label', filename)
        if os.path.exists(exist_filepath):
            new_filepath = os.path.join('data', folder,'label',filename)
            os.replace(exist_filepath, new_filepath)
'''

img =cv2.imread(os.path.join('data','train','image','1a128f16-05ed-11ee-a24e-e0d4e8f60c6a.jpg'))
print(img.shape) #picture shape(480,640,3)

# Dieser Code wird von Albumentationsbibliothek importiert, der mit Hilfe einer Transformationspipeline aus Albumentationsbibliothek definiert wird.
# Er wird von diesem Link (https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/) genommen.
transform = A.Compose([
            A.RandomCrop(width=450, height=450),#crop the foto to 450 pixel width and Height
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2)],
            bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))

#Hier wird eine Schleife über verschiedene Teildatensätze (test, train, validate) durchlaufen lassen und zwischen jeder Schleife werden die Fotos im entsprechenden Unterordner gelesen.
#Für jedes Foto wird eine Bildaugmentation durchgeführt. Die Koordinaten der Bounding Box werden aus einer Label-Datei extrahiert und normalisiert. Dann werden 50 augmentierte Versionen des Fotos generiert.
#Jedes augmentierte Foto wird gespeichert und eine entsprechende Annotation wird in einer JSON-Datei erstellt. Die Annotation enthält Informationen über das Bild, die Bounding Box-Koordinaten und die Klassenzuordnung.
#Für Diesen Code wird die zuvor definierte Transformationspipeline verwendet, die die Albumentationsfunktionen enthält.
"""
#pipeline
#Schleife über verschiedene Teildatensätze(test, train, validate)
for part in ['test','train','validate']:
    # Schleife über Fotos im entsprechenden Teilordner
    for fotos in os.listdir(os.path.join('data', part, 'image')):
        foto = cv2.imread(os.path.join('data', part, 'image', fotos))
        
        coordinate = [0,0,0.001,0.001]
        label_path = os.path.join('data', part, 'label', f'{fotos.split(".")[0]}.json')
        # Überprüfen, ob eine Label-Datei für das Foto existiert
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coordinate[0] = label['shapes'][0]['points'][0][0]
            coordinate[1] = label['shapes'][0]['points'][0][1]
            coordinate[2] = label['shapes'][0]['points'][1][0]
            coordinate[3] = label['shapes'][0]['points'][1][1]
            coordinate = list(np.divide(coordinate, [640,480,640,480]))

            try:
                for x in range(50):
                    augmented = transform(image=foto, bboxes=[coordinate], class_labels=['face'])
                    cv2.imwrite(os.path.join('augmented_data', part, 'image', f'{fotos.split(".")[0]}.{x}.jpg'),
                            augmented['image'])

                    annotation = {}
                    annotation['image'] = fotos

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0
                        else:
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0

                    with open(os.path.join('augmented_data', part, 'label', f'{fotos.split(".")[0]}.{x}.json'), 'w') as f:
                        json.dump(annotation, f)

            except Exception as e:
                print(e)
"""
# Alle Jpg-Dateien werden in TensorFlow-Datensätze konvertiert, um die Vorverarbeitungsschritte für Trainings-, Validierungs- und Testbilder durchzuführen.
#Anschließend wird die Funktion load_image auf jedes Bild angewendet, um das Bild in das gewünschte Format zu laden.
#Dann wird jedes Bild auf Größe 120x120 Pixel geändert, indem die Funktion tf.image.resize verwendet wird.
#Zur Normalisierung werden die Pixelwerte jedes Bildes durch 255 geteilt.
#test image(ordner)
test_image = tf.data.Dataset.list_files("augmented_data\\test\\image\\*.jpg",shuffle=False)
test_image = test_image.map(load_image)
test_image = test_image.map(lambda x: tf.image.resize(x, (120,120)))
test_image = test_image.map(lambda x: x/255)

#train image
train_image = tf.data.Dataset.list_files("augmented_data\\train\\image\\*.jpg",shuffle=False)
train_image = train_image.map(load_image)
train_image = train_image.map(lambda x: tf.image.resize(x, (120,120)))
train_image = train_image.map(lambda x: x/255)

#validate image
validate_image = tf.data.Dataset.list_files("augmented_data\\validate\\image\\*.jpg",shuffle=False)
validate_image = validate_image.map(load_image)
validate_image = validate_image.map(lambda x: tf.image.resize(x, (120,120)))
validate_image = validate_image.map(lambda x: x/255)
"""
#try
next_image = next(train_image.as_numpy_iterator())
print(next_image)
"""

#label loading function
def loading_label(label_path):
    with open(label_path.numpy(), "r", encoding="utf-8") as f:
        label = json.load(f)

    return [label["class"]], label["bbox"]

# Alle Json-Dateien aus Label-Ordner werden in Tensorflow-Datensätze importiert.
test_label = tf.data.Dataset.list_files("augmented_data\\test\\label\\*.json",shuffle=False)
test_label = test_label.map(lambda x: tf.py_function(loading_label, [x], [tf.uint8, tf.float16]))
print(test_label)
train_label = tf.data.Dataset.list_files("augmented_data\\train\\label\\*.json",shuffle=False)
train_label = train_label.map(lambda x: tf.py_function(loading_label, [x], [tf.uint8, tf.float16]))

validate_label = tf.data.Dataset.list_files("augmented_data\\validate\\label\\*.json",shuffle=False)
validate_label = validate_label.map(lambda x: tf.py_function(loading_label, [x], [tf.uint8, tf.float16]))
"""
#test is it loaded or not
next_image = next(test_label.as_numpy_iterator())
print(next_image)
"""

#Hier werden image(jpg Datei) und label(Json Datei) als zip Datei kombiniert
test = tf.data.Dataset.zip((test_image, test_label))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

train = tf.data.Dataset.zip((train_image, train_label))
train = train.shuffle(1300)
train = train.batch(8)
train = train.prefetch(4)

validate = tf.data.Dataset.zip((validate_image, validate_label))
validate = validate.shuffle(1300)
validate = validate.batch(8)
validate = validate.prefetch(4)

shape = train.as_numpy_iterator().next()[0].shape
print(shape)

data_samples = test.as_numpy_iterator()
res = data_samples.next()
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

#cv2 und matplotlib werden hier importiert. In diesem Fall zeigen sich 4 Bilder mit roter Makierung in Gesicht, deren Datensätze gesammelt werden.
for index in range(4):
    sample_image = res[0][index]
    sample_coords = res[1][1][index]

    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                  (255, 0, 0), 2)

    ax[index].imshow(sample_image)

plt.show()

#Build Neural Network
#Hier wird Convolutional Neurale Netze heruntergeladen und diese Gesichtserkennung wird von VGG16 benutzt.
"""
vgg = VGG16(include_top = False)
vgg.summary()
"""

# Mit der Funktion "build_model()" wird ein Model für die Gesichtserkennung erstellt, das von der VGG16-Architektur als Basis verwendet wird.
#Das Ausgabemodel besteht aus zwei Teilen.
#Zur Klassifizierung des Models und dem Zurücksetzen der Wahrscheinlichkeit werden im ersten Teil "GlobalMaxPooling2D" und VGG16-Architektur(zur Ausgabe) angewendet.
#Der zweite Teil wird analog wie erster Teil aufgebaut, welches zur Regression dient und die Koordinaten eines begrenzenden Rechtecks zurücksetzen, die das erkannte Gesicht umgibt.

def build_model():
    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)

    c1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(c1)
    class2 = Dense(1, activation='sigmoid')(class1)

    r1 = GlobalMaxPooling2D()(vgg)
    reg1 = Dense(2048, activation='relu')(r1)
    reg2 = Dense(4, activation='sigmoid')(reg1)

    facerecog = Model(inputs=input_layer, outputs=[class2, reg2])
    return facerecog


facerecog = build_model()
"""
a = facerecog.summary()
print(a)
"""

#print(len(train)) #650 data
#In diesem Code-Ausschnitt wird die Variable epoch bestimmt, indem die Länge des train-Objekts ermittelt wird.
# Anschließend wird der learningrate_decay berechnet, indem 1 durch 0,75 dividiert und daraus 1 subtrahiert wird.
# Das Ergebnis wird dann durch die Anzahl der Epochen geteilt. Schließlich wird der berechnete Wert von learningrate_decay auf der Konsole ausgegeben.
#Das Learningrate beträgt 0.0005128205128205127, die als niedrig betrachtet werden kann.
epoch = len(train)
learningrate_decay = (1./0.75-1)/epoch
print("learningrate:", learningrate_decay)

#Aus der tensorflow-Bibliothek wird ein Code-Ausschnitt importiert. Anschließend wird das Learningrate mit dem Optimierer "Adam" definiert und der Variable optimiert zugewiesen.
#Der Adam-Optimierer wird mit einem Lernrate von 0.0005 und dem angegebenen Lernratenabfall (learningrate_decay) konfiguriert.
#Der Optimierer wird verwendet, um die Modellparameter während des Trainingsprozesses zu aktualisieren und die Verlustfunktion zu minimieren.
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005, decay=learningrate_decay)


# Hier wird Mathematik Formel benutzt, die aus diesem Link(https://stats.stackexchange.com/questions/319243/object-detection-loss-function-yolo) genommen wird, welches zur Berechnung classloss und regressloss dient.
# classloss and regressloss
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]

    h_pred = yhat[:, 3] - yhat[:, 1]
    w_pred = yhat[:, 2] - yhat[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size

classloss = tf.keras.losses.BinaryCrossentropy()
regloss = localization_loss


#Dieser Code wird neurale Netze nachtrainiert.
#train neural network pipeline
class FaceTracker(Model):
    def __init__(self, tracker, **kwargs):
        super().__init__(**kwargs)
        self.model = tracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = optimizer

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

model = FaceTracker(facerecog)
model.compile(optimizer, classloss, regloss)

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#In diesem Code-Ausschnitt wird das Model mit den Trainingsdaten für 2 Epochen trainiert.
#Der TensorBoard-Callback wird verwendet, um die Trainingsmetriken aufzuzeichnen. Das Ergebnis des Trainings wird in der Variable "hist" gespeichert.
#Pro Epoch wird ca. 13 min dauern. Zuerst werden 30 Epochen für ca. 6-7 Stunden und danach noch 5 Epochen trainert, leider hat das nicht geklappt.
#Anschließend werden 2 Epochen(in facerecog.h5 gespeichert) erfolgreich trainiert
"""
#hist = model.fit(train, epochs=2, validation_data=validate, callbacks=[tensorboard_callback],)
#facerecog.save('facerecog.h5')
"""
#Die Datei(facerecog.h5) wird in variable facerecog geladen.
facerecog = load_model('facerecog.h5')

#Zur Gesichtserkennung wird mit Hilfe diesem Code das Kamera angeschaltet.
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500,:]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facerecog.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [80, 0])),
                      (255, 0, 0), -1)

        # Controls the text rendered
        cv2.putText(frame, 'Gesicht', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('FaceTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#Quellen:
#Albumentation: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
#Classloss und Regressloss: https://stats.stackexchange.com/questions/319243/object-detection-loss-function-yolo
#Idee: https://www.youtube.com/watch?v=N_W4EYtsa10&t=6643s&ab_channel=NicholasRenotte