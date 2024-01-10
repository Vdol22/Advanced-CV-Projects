import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


set_background('./bg/bg.png')

# Название вкладки
st.title('Pneumonia classification')

# Заголовок
st.header('Please upload a chest X-ray image')

# Окно загрузки файла
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Модель инференса
model = load_model('./model/pneumonia_classifier.h5')

# Имена классов
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# Вывод изображения
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Инференс
    class_name, conf_score = classify(image, model, class_names)

    # Вывод результата
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
