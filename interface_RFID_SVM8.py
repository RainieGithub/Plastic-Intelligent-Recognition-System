import Tkinter as tk
import tkMessageBox
import ttk
from PIL import Image, ImageTk
import cv2
import sys
import numpy as np
from sklearn.externals import joblib
#sys.modules['sklearn.externals.joblib'] = joblib
#import joblib
from csv import reader
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D 
import time
import math
import os
import serial
import ssl
import urllib
import pickle
import matebdTk2
from NIRS import NIRS

ssl._create_default_https_context = ssl._create_unverified_context
nm_array = np.arange(3000)
data_fi2 = np.zeros(3000)
k = 3

cap = cv2.VideoCapture(0)
width, height = 515, 386
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, float(0.001))

pca5 = joblib.load('/home/ispect/Desktop/0302/0309/libs/20230414pca.m')
clf5 = joblib.load('/home/ispect/Desktop/0302/0309/libs/20230414clf.pkl')


pca6 = joblib.load('/home/ispect/Desktop/0302/0309/libs/0519cup_PP_PVC_clf.m')
clf6 = joblib.load('/home/ispect/Desktop/0302/0309/libs/0519cup_PP_PVC_pca.pkl')

TC_use_m = joblib.load('/home/ispect/Desktop/0302/0309/libs/0509cloth_TC24568_1.m')
TC_use_clf = joblib.load('/home/ispect/Desktop/0302/0309/libs/0509cloth_TC24568_1.pkl')

nirs = NIRS()
nirs.set_config(8, NIRS.TYPES.HADAMARD_TYPE, 86, 6, 950, 1700, 17)
nirs.set_pga_gain(new_value=0)
nirs.scan_snr("hadamard")

def PCA_detectionWindows():
    def execute():
        progressbarTwo['maximum'] = 9
        progressbarTwo['value'] = 0
        Enter_button['state'] = 'disable'
        END_button['state'] = 'disable'
        shutdown_button['state'] = 'disable'
        progressbarTwo['value'] += 1
        PCA_windows.update_idletasks()
        
        pca = joblib.load('/home/ispect/Desktop/0302/0309/libs/20230628bottle7category.m')
        clf2 = joblib.load('/home/ispect/Desktop/0302/0309/libs/20230628bottle7category.pkl')
        
        print("Scanning...")
        nirs.scan()
        results2 = nirs.get_scan_results()
        wave = results2["wavelength"]
        intensity = results2["intensity"]  # original raw intensity
        progressbarTwo['value'] += 1
        PCA_windows.update_idletasks()
        
        intensityf = open('/home/ispect/Desktop/0302/0309/root_emmc/root_ref_intensity.txt', 'r')
        reference = intensityf.read().split(", ")
        print(reference, type(reference), len(reference))
        wave = wave
        Ref = reference
        Sam = intensity
        temp_final = []
        final = []
        sim = intensity
        progressbarTwo['value'] += 1
        PCA_windows.update_idletasks()
        
        for i in range(0, len(wave)):
            if int(Ref[i]) == 0:
                Ref[i] = 1
            sim[i] = float(Sam[i]) / float(Ref[i])
            if sim[i] >= 1:
                sim[i] = 0.999
            elif sim[i] == 0:
                sim[i] = 0.001
            temp_final.append(-(math.log(sim[i], 10)))
        
        progressbarTwo['value'] += 1
        PCA_windows.update_idletasks()
        
        for i in range(0, len(wave)):
            final.append((temp_final[i] - min(temp_final)) / (max(temp_final) - min(temp_final)))
        
        test_100_fi_nm = wave
        test_100_fi_data = final
        data_fi2 = np.zeros(3000)
        
        for i in range(0, len(test_100_fi_data)):
            for j in range(0, 3000):
                if int(test_100_fi_nm[i]) == nm_array[j]:
                    data_fi2[j] = test_100_fi_data[i]
        
        progressbarTwo['value'] += 1
        PCA_windows.update_idletasks()
        
        data_fi2 = np.array(data_fi2).reshape(1, len(data_fi2))
        test_data_fi = pca.transform(data_fi2)
        
        progressbarTwo['value'] += 1
        PCA_windows.update_idletasks()
        
        pred = clf2.predict(test_data_fi)
        result = str(pred)
        
        pred_proba = clf2.predict_proba(test_data_fi)
        pred_classes_ = clf2.classes_
        print("result:" + str(pred))
        print("proba:" + str(pred_proba))
        print("pred_classes_:" + str(pred_classes_))
        pred_class_num = -1
        
        for i in range(0, len(pred_classes_)):
            if pred_classes_[i] == pred:
                pred_class_num = i
                break
        
        print("pred_n = ", pred_proba[0, pred_class_num])
        
        if pred_proba[0, pred_class_num] < 0.2:
            result = "['Others']"
        
        print("sample predict = " + result)
        
        progressbarTwo['value'] += 1
        PCA_windows.update_idletasks()
        
        test_data_fi6 = pca6.transform(data_fi2)
        pred6 = clf6.predict(test_data_fi6)
        result_PVC = str(pred6)
        
        pred_proba6 = clf6.predict_proba(test_data_fi6)
        pred_classes_6 = clf6.classes_
        print("result6:" + str(pred6))
        print("proba6:" + str(pred_proba6))
        print("pred_classes_6:" + str(pred_classes_6))
        
        pred_class_num6 = -1
        
        for i in range(0, len(pred_classes_6)):
            if pred_classes_6[i] == pred6:
                pred_class_num6 = i
                break
        
        print("pred_n6 = ", pred_proba6[0, pred_class_num6])
        
        UPUP = 0
        
        if result == "['GPPS']":
            UPUP = 10
            print("Hello world")
            result == "['PS']"
        elif result == "['HIPS']":
            UPUP = 20
            print("Yabe!Yabe!")
            result == "['PS']"
        elif result == "['LDPE']":
            UPUP = 30
            print('milk')
        elif result == "['HDPE']":
            UPUP = 31
            print('paper cup')
        elif result == "['PP']":
            UPUP = 32
        elif result == "['PVC']":
            UPUP = 33
        elif result == "['PET']":
            UPUP = 34
        elif result == "['PLA']":
            UPUP = 35
        elif result == "['PE and Paper']":
            UPUP = 36
        else:
            UPUP = 0
        
        progressbarTwo['value'] += 1
        PCA_windows.update_idletasks()
        
        UUF = open('/home/ispect/Desktop/0302/0309/libs/interface_category.txt', 'r')
        UUP = UUF.read()
        UUP = int(UUP)
        print(UUP, type(UUP))
        
        progressbarTwo['value'] += 1
        PCA_windows.update_idletasks()
        
        UUP = UUP + UPUP
        UUF = open('/home/ispect/Desktop/0302/0309/libs/interface_category.txt', 'w')
        UUP = str(UUP)
        UUF.write(UUP)
        UUF.close()
        
        progressbarTwo['value'] = 0
        PCA_windows.update_idletasks()
        
        Enter_button['state'] = 'normal'
        END_button['state'] = 'normal'
        shutdown_button['state'] = 'normal'
        back_button['state'] = 'normal'
        
        with open('/home/ispect/Desktop/0302/0309/libs/interface_bottle_cups.txt') as csvfile:
            test_100_fi_data = np.loadtxt(csvfile, delimiter=",")
            print(test_100_fi_data, type(test_100_fi_data), len(test_100_fi_data))
            test_100_fi_data = np.array(test_100_fi_data).reshape(1, len(test_100_fi_data))
            test_data_fi = pca5.transform(test_100_fi_data)
            pred2 = clf5.predict(test_data_fi)
            bottlecup_result = str(pred2)
            
            print("modelresult:" + str(pred2))
            
            if result == "['PET']" and bottlecup_result == "['CUP']":
                my_var_2.set("Bottle")
            elif result == "['PET']" and bottlecup_result == "['Bottle']":
                my_var_2.set("Bottle")
            elif result == "['PP']" and bottlecup_result == "['CUP']":
                my_var_2.set("cup")
            elif result == "['PP']" and bottlecup_result == "['Bottle']":
                my_var_2.set("cup")
            elif result == "['HDPE']" and bottlecup_result == "['Bottle']":
                my_var_2.set("Bottle")
            elif result == "['HDPE']" and bottlecup_result == "['CUP']":
                my_var_2.set("Bottle")
            elif result == "['PE']" and bottlecup_result == "['CUP']":
                my_var_2.set("PE")
            elif result == "['PE']" and bottlecup_result == "['Bottle']":
                my_var_2.set("PE")
            elif result == "['LDPE']":
                my_var_2.set("LDPE")
            elif result == "['PVC']":
                my_var_2.set("PVC")
            elif result == "['PE and Paper']":
                my_var_2.set("PE and Paper")
            elif result == "['GPPS']":
                my_var_2.set("PS")
            elif result == "['HIPS']":
                my_var_2.set("PS")
            elif result == "['PLA']":
                my_var_2.set("PLA")
            elif result == "['Others']" and (bottlecup_result == "['Bottle']" or bottlecup_result == "['CUP']"):
                my_var_2.set("Others")
            else:
                my_var_2.set("Others")
            
            if result_PVC == "['PVC']":
                my_var_2.set("Others")
                result = result_PVC
            my_var_1.set(str(UPUP))  # uupon
            my_var_3.set(UUP)  # total uupon
            my_var_4.set(result)  # material result

    def execute_2():
        PCA_windows.destroy()
        print("a")
        Setting2()

    def uploading_google():
        UUF = open('/home/ispect/Desktop/0302/0309/libs/interface_category.txt', 'r')
        UUP = UUF.read()
        print(UUP, type(UUP))

        url = ('https://script.google.com/macros/s/AKfycbw_2TQC92xPbuT27bhriqgk7Y_4bN-W6mgxCOLTB9ktIMTxRqsaDipNj0jcIGYe46F5aw/exec?data=')
        urllib.urlretrieve(url + str(UID) + "," + str(UUP))
        print("Hello world")

        UUF = open('/home/ispect/Desktop/0302/0309/libs/interface_category.txt', 'w')
        UUP = str(0)
        UUF.write(UUP)
        UUF.close()

        tkMessageBox.showinfo('warning', "Uploading UID and UUPON successfully")
        PCA_windows.destroy()
        print("uploading successfully")

        RFID_CODE()

    def back():
        PCA_windows.destroy()
        choosen()

    def open_camera():
        ret, frame = cap.read()
        roi_frame = frame[150:380, 50:550]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        ret, output = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1, 1), dtype=np.uint8)
        erosion = cv2.erode(output, kernel, iterations=1)
        hc1 = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        con1 = hc1[1]
        n1 = len(con1)

        for c in con1:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 209 and w < 490 and h > 60:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame[150:480, 50:550], [box], 0, (255, 255, 255), 2)
                angle = np.arctan((float(h) / 2) / (float(w) / 4))
                w_mm = w * 0.54054054
                h_mm = h * 0.50847458

                cv2.putText(frame, "length = " + str(round(w_mm)) + " mm", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "width = " + str(round(h_mm)) + " mm", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "angle = " + str(angle), (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                with open("/home/ispect/Desktop/0302/0309/libs/interface_bottle_cups.txt", "w") as inform:
                    inform.write(str(w_mm) + "," + str(h_mm) + "," + str(angle) + "\n")

        opencv_image = frame
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        #label_widget.photo_image = photo_image
        #label_widget.configure(image=photo_image)
        #label_widget.after(1, open_camera)

    RFUID = open('/home/ispect/Desktop/0302/0309/libs/interface_UID.txt', 'r')
    UID = RFUID.read()
    print(UID, type(UID))

    if UID == "C55CD86B":
        data_name = str("Yu-Shing Zhou")
    elif UID == "451B3749":
        data_name = str("Jungle Chen")
    elif UID == "9536C36B":
        data_name = str("Yu-Ping Chen")
    elif UID == "F585CB6B":
        data_name = str("Ya-Pei Chou")
    elif UID == "251B96B":
        data_name = str("Hsin-I Ho")
    elif UID == "75CDAB28":
        data_name = str("Jen-Jie Chieh")
    else:
        data_name = str("NO data")

    PCA_windows = tk.Tk()
    PCA_windows.title('Enter filename and condition')
    PCA_windows.attributes("-fullscreen", True)
    #PCA_windows.geometry('1024x600+0+0')
    PCA_windows.configure(bg="lightsteelblue")

    my_var_1 = tk.StringVar(PCA_windows)
    my_var_1.set("NONE")

    my_var_2 = tk.StringVar(PCA_windows)
    my_var_2.set("NONE")

    my_var_3 = tk.StringVar(PCA_windows)
    my_var_3.set("NONE")

    my_var_4 = tk.StringVar(PCA_windows)
    my_var_4.set("NONE")

    img3 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/newmaterial2.png')
    img3 = img3.resize((900, 400), Image.ANTIALIAS)
    tk_img_3 = ImageTk.PhotoImage(img3)

    img4 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/tebu.png')
    img4 = img4.resize((300, 84), Image.ANTIALIAS)
    tk_img_4 = ImageTk.PhotoImage(img4)

    img5 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/upload.png')
    img5 = img5.resize((240, 77), Image.ANTIALIAS)
    tk_img_5 = ImageTk.PhotoImage(img5)

    img6 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/back.png')
    img6 = img6.resize((80, 80), Image.ANTIALIAS)
    tk_img_6 = ImageTk.PhotoImage(img6)

    labelD = tk.Label(PCA_windows, image=tk_img_3, width=900, height=400)
    labelD.config(bg="lightsteelblue")
    labelD.place(x=0, y=10)

    UID_check = tk.Label(PCA_windows, text="Name = " + data_name + "   UID = " + UID)
    UID_check.config(font=("Times New Roman", 18, "bold"), bg="lightsteelblue")
    UID_check.place(x=500, y=550)

    PCA_predict = tk.Label(PCA_windows, textvariable=my_var_2)
    PCA_predict.config(font=("Times New Roman", 40, "bold"), bg="lightsteelblue")
    PCA_predict.place(x=300, y=35)

    UUPON = tk.Label(PCA_windows, textvariable=my_var_4)
    UUPON.config(font=("Times New Roman", 40, "bold"), bg="lightsteelblue")
    UUPON.place(x=300, y=145)

    Total_uupon = tk.Label(PCA_windows, textvariable=my_var_1)
    Total_uupon.config(font=("Times New Roman", 28, "bold"), bg="lightsteelblue")
    Total_uupon.place(x=300, y=250)

    Material = tk.Label(PCA_windows, textvariable=my_var_3)
    Material.config(font=("Times New Roman", 28, "bold"), bg="lightsteelblue")
    Material.place(x=300, y=330)

    progressbarTwo = ttk.Progressbar(PCA_windows, mode="determinate")
    progressbarTwo.place(x=228, y=435, width=500, height=10)

    Enter_button = tk.Button(PCA_windows, image=tk_img_4, command=execute)
    Enter_button.config(font=("Times New Roman", 18, "bold"), bg="white")
    Enter_button.place(x=140, y=447, width=350, height=84)

    END_button = tk.Button(PCA_windows, image=tk_img_5, command=uploading_google)
    END_button.config(font=("Times New Roman", 18, "bold"), bg="white")
    END_button.place(x=510, y=447, width=350, height=84)

    shutdown_button = tk.Button(PCA_windows, text='Renew Measure Reference', command=execute_2)
    shutdown_button.config(font=("Times New Roman", 9, "bold"), bg="lightsteelblue")
    shutdown_button.place(x=5, y=557, width=200, height=18)

    back_button = tk.Button(PCA_windows, image=tk_img_6, command=back)
    back_button.config(font=("Times New Roman", 18, "bold"), bg="white")
    back_button.place(x=10, y=447, width=84, height=84)

    PCA_windows.mainloop()


def QAQ_detectionWindows():
    def execute():
        Enter_button['state'] = 'disable'
        END_button['state'] = 'disable'
        shutdown_button['state'] = 'disable'

        progressbarOne['maximum'] = 9
        progressbarOne['value'] = 0

        pca = joblib.load('/home/ispect/Desktop/0302/0309/libs/0509cloth_PET_N_RA_C_N_1.m')
        clf2 = joblib.load('/home/ispect/Desktop/0302/0309/libs/0509cloth_PET_N_RA_C_N_1.pkl')

        progressbarOne['value'] += 1
        QAQ_windows.update()

        from NIRS import NIRS

        progressbarOne['value'] += 1
        QAQ_windows.update()

        print("Scanning...")
        nirs.scan()
        results2 = nirs.get_scan_results()
        wave = results2["wavelength"]
        intensity = results2["intensity"]  # original raw intensity

        intensityf = open('/home/ispect/Desktop/0302/0309/root_emmc/root_ref_intensity.txt', 'r')
        reference = intensityf.read().split(", ")
        print(reference, type(reference), len(reference))
        wave = wave
        Ref = reference
        Sam = intensity
        temp_final = []
        final = []
        sim = intensity

        progressbarOne['value'] += 1
        QAQ_windows.update()
        for i in range(0, len(wave)):
            if int(Ref[i]) == 0:
                Ref[i] = 1
            sim[i] = float(Sam[i]) / float(Ref[i])
            if sim[i] >= 1:
                sim[i] = 0.999
            elif sim[i] == 0:
                sim[i] = 0.001
            temp_final.append(-(math.log(sim[i], 10)))

        progressbarOne['value'] += 1
        QAQ_windows.update()
        for i in range(0, len(wave)):
            final.append((temp_final[i] - min(temp_final)) / (max(temp_final) - min(temp_final)))

        progressbarOne['value'] += 1
        QAQ_windows.update()

        test_100_fi_nm = wave
        test_100_fi_data = final
        print(test_100_fi_nm)
        print(test_100_fi_data)
        data_fi2 = np.zeros(3000)
        for i in range(0, len(test_100_fi_data)):
            for j in range(0, 3000):
                if int(test_100_fi_nm[i]) == nm_array[j]:
                    data_fi2[j] = test_100_fi_data[i]

        progressbarOne['value'] += 1
        QAQ_windows.update()

        data_fi2 = np.array(data_fi2).reshape(1, len(data_fi2))
        test_data_fi = pca.transform(data_fi2)

        pred = clf2.predict(test_data_fi)

        pred_proba = clf2.predict_proba(test_data_fi)
        pred_classes_ = clf2.classes_
        print("result:" + str(pred))
        print("proba:" + str(pred_proba))
        print("pred_classes_:" + str(pred_classes_))
        pred_class_num = -1
        for i in range(0, len(pred_classes_)):
            if pred_classes_[i] == pred:
                pred_class_num = i
                break
        result = str(pred)

        print("pred_n = ", pred_proba[0, pred_class_num])
        if pred_proba[0, pred_class_num] < 0.25 and result != "['C']" and result != "['PET40%Cotton60%']" and result != "['PET50%Cotton50%']" and result != "['PET60%Cotton40%']":
            result = 'Others'
        print("sample predict = " + result)

        progressbarOne['value'] += 1
        QAQ_windows.update()
        if result == "['T']" or result == "['C']" or result == "['PET40%Cotton60%']" or result == "['PET50%Cotton50%']" or result == "['PET60%Cotton40%']":
            test_data_fi_pp_use = TC_use_m.transform(data_fi2)
            TC_pred = TC_use_clf.predict(test_data_fi_pp_use)
            TC_pred = str(TC_pred)
            print("TC use:", TC_pred)
            if TC_pred == "['T']":
                result = "['T']"
            elif TC_pred == "['C']":
                result = "['C']"
            elif TC_pred == "['PET20%Cotton80%']":
                result = "['C']"
            elif TC_pred == "['PET40%Cotton60%']":
                result = "['PET40%Cotton60%']"
            elif TC_pred == "['PET50%Cotton50%']":
                result = "['PET50%Cotton50%']"
            elif TC_pred == "['PET60%Cotton40%']":
                result = "['PET60%Cotton40%']"
            elif TC_pred == "['PET80%Cotton20%']":
                result = "['T']"
            else:
                result = "['T']"

        progressbarOne['value'] += 1
        QAQ_windows.update()

        UPUP = 0
        if result == "['T']":
            UPUP = 10
            my_var_4.set("PET")
        elif result == "['Nylon']":
            UPUP = 20
            my_var_4.set("Nylon")
        elif result == "['R']":
            UPUP = 30
            my_var_4.set("Rayon")
        elif result == "['C']":
            UPUP = 40
            my_var_4.set("Cotton")
        elif result == "['TPU']":
            UPUP = 50
            my_var_4.set("TPU")
        elif result == "['W']":
            UPUP = 60
            my_var_4.set("Wool")
        elif result == "['Ace']":
            UPUP = 70
            my_var_4.set("Ace")
        elif result == "['Acrylic']":
            UPUP = 80
            my_var_4.set("Acrylic")
        elif result == "['PET20%Cotton80%']":
            UPUP = 1
            my_var_4.set("PET20% Cotton80%")
        elif result == "['PET40%Cotton60%']":
            UPUP = 2
            my_var_4.set("PET40% Cotton60%")
        elif result == "['PET50%Cotton50%']":
            UPUP = 3
            my_var_4.set("PET50% Cotton50%")
        elif result == "['PET60%Cotton40%']":
            UPUP = 4
            my_var_4.set("PET60% Cotton40%")
        elif result == "['PET80%Cotton20%']":
            UPUP = 5
            my_var_4.set("PET80% Cotton20%")
        elif result == "['PP']":
            UPUP = 6
            my_var_4.set("PP")
        elif result == "['HDPE']":
            UPUP = 7
            my_var_4.set("PE")
        else:
            UPUP = 0
            my_var_4.set("Others")

        progressbarOne['value'] += 1
        QAQ_windows.update()

        UUF = open('/home/ispect/Desktop/0302/0309/libs/interface_category.txt', 'r')
        UUP = UUF.read()
        UUP = int(UUP)
        print(UUP, type(UUP))

        UUP = UUP + UPUP
        UUF = open('/home/ispect/Desktop/0302/0309/libs/interface_category.txt', 'w')
        UUP = str(UUP)
        UUF.write(UUP)
        UUF.close()

        my_var_1.set("+" + str(UPUP))
        my_var_3.set(UUP)

        progressbarOne['value'] += 1
        QAQ_windows.update()

        Enter_button['state'] = 'normal'
        END_button['state'] = 'normal'
        shutdown_button['state'] = 'normal'
        back_button['state'] = 'normal'

        progressbarOne['value'] = 0
        QAQ_windows.update()

    def execute_2():
        QAQ_windows.destroy()
        print("a")
        Setting2()

    def uploading_google():
        UUF = open('/home/ispect/Desktop/0302/0309/libs/interface_category.txt', 'r')
        UUP = UUF.read()
        print(UUP, type(UUP))

        url = (
            'https://script.google.com/macros/s/AKfycbw_2TQC92xPbuT27bhriqgk7Y_4bN-W6mgxCOLTB9ktIMTxRqsaDipNj0jcIGYe46F5aw/exec?data=')
        urllib.urlretrieve(url + str(UID) + "," + str(UUP))
        print("Hello world")

        UUF = open('/home/ispect/Desktop/0302/0309/libs/interface_category.txt', 'w')
        UUP = str(0)
        UUF.write(UUP)
        UUF.close()

        tkMessageBox.showinfo('warning', "Uploading UID and UUPON successfully")
        QAQ_windows.destroy()
        print("uploading successfully")

        RFID_CODE()

    def back():
        QAQ_windows.destroy()
        choosen()

    RFUID = open('/home/ispect/Desktop/0302/0309/libs/interface_UID.txt', 'r')
    UID = RFUID.read()

    print(UID, type(UID))

    if UID == "C55CD86B":
        data_name = str("Yu-Shing Zhou")
    elif UID == "451B3749":
        data_name = str("Jungle Chen")
    elif UID == "9536C36B":
        data_name = str("Yu-Ping Chen")
    elif UID == "F585CB6B":
        data_name = str("Ya-Pei Chou")
    elif UID == "251B96B":
        data_name = str("Hsin-I Ho")
    elif UID == "75CDAB28":
        data_name = str("Jen-Jie Chieh")
    else:
        data_name = str("NO data")

    QAQ_windows = tk.Tk()
    QAQ_windows.title('Enter filename and condition')
    QAQ_windows.attributes("-fullscreen", True)
    #QAQ_windows.geometry('1024x600+0+0')
    QAQ_windows.configure(bg="lightsteelblue")

    my_var_1 = tk.StringVar(QAQ_windows)
    my_var_1.set("NONE")

    my_var_3 = tk.StringVar(QAQ_windows)
    my_var_3.set("NONE")

    my_var_4 = tk.StringVar(QAQ_windows)
    my_var_4.set("NONE")

    img3 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/newmaterial3.png')
    img3 = img3.resize((900, 400), Image.ANTIALIAS)
    tk_img_3 = ImageTk.PhotoImage(img3)

    img4 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/clbu.png')
    img4 = img4.resize((300, 84), Image.ANTIALIAS)
    tk_img_4 = ImageTk.PhotoImage(img4)

    img5 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/upload.png')
    img5 = img5.resize((240, 77), Image.ANTIALIAS)
    tk_img_5 = ImageTk.PhotoImage(img5)

    img6 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/back.png')
    img6 = img6.resize((80, 80), Image.ANTIALIAS)
    tk_img_6 = ImageTk.PhotoImage(img6)

    labelD = tk.Label(QAQ_windows, image=tk_img_3, width=900, height=400)
    labelD.config(bg="lightsteelblue")
    labelD.place(x=0, y=10)

    UID_check = tk.Label(QAQ_windows, text="Name = " + data_name + "   UID = " + UID)
    UID_check.config(font=("Times New Roman", 18, "bold"), bg="lightsteelblue")
    UID_check.grid(row=0, column=0)
    UID_check.grid(padx=500, pady=550, sticky='nw')

    UUPON = tk.Label(QAQ_windows, textvariable=my_var_4)
    UUPON.config(font=("Times New Roman", 60, "bold"), bg="lightsteelblue")
    UUPON.grid(row=0, column=0)
    UUPON.grid(padx=150, pady=110, sticky='nw')

    Total_uupon = tk.Label(QAQ_windows, textvariable=my_var_1)
    Total_uupon.config(font=("Times New Roman", 28, "bold"), bg="lightsteelblue")
    Total_uupon.grid(row=0, column=0)
    Total_uupon.grid(padx=400, pady=255, sticky='nw')

    Material = tk.Label(QAQ_windows, textvariable=my_var_3)
    Material.config(font=("Times New Roman", 28, "bold"), bg="lightsteelblue")
    Material.grid(row=0, column=0)
    Material.grid(padx=400, pady=335, sticky='nw')

    Enter_button = tk.Button(QAQ_windows, image=tk_img_4, command=execute)
    Enter_button.grid(row=5, column=1)
    Enter_button.config(font=("Times New Roman", 18, "bold"), bg="white")
    Enter_button.place(x=140, y=447, width=350, height=84)

    progressbarOne = ttk.Progressbar(QAQ_windows)
    progressbarOne.place(x=228, y=435, width=500, height=10)

    END_button = tk.Button(QAQ_windows, image=tk_img_5, command=uploading_google)
    END_button.grid(row=5, column=1)
    END_button.config(font=("Times New Roman", 18, "bold"), bg="white")
    END_button.place(x=510, y=447, width=350, height=84)

    shutdown_button = tk.Button(QAQ_windows, text='Renew Measure Reference', command=execute_2)
    shutdown_button.grid(row=5, column=1)
    shutdown_button.config(font=("Times New Roman", 9, "bold"), bg="lightsteelblue")
    shutdown_button.place(x=5, y=557, width=200, height=18)

    back_button = tk.Button(QAQ_windows, image=tk_img_6, command=back)
    back_button.grid(row=5, column=1)
    back_button.config(font=("Times New Roman", 18, "bold"), bg="white")
    back_button.place(x=10, y=447, width=84, height=84)

    QAQ_windows.mainloop()

def Setting2():
    def judge_event():
        progressbarOne['maximum'] = 9
        progressbarOne['value'] = 0
        if k == 1:
            tkMessageBox.showwarning('warning', 'double test')
        elif k == 2:
            tkMessageBox.showerror('warning', 'double test')
        else:
            from NIRS import NIRS

            start = time.time()
            pca = joblib.load('/home/ispect/Desktop/0302/0309/libs/QAQ_pca.m')
            nm_array = np.arange(3000)
            data_fi2 = np.zeros(3000)

            progressbarOne['value'] += 1
            Input_windows.update()

            progressbarOne['value'] += 1
            Input_windows.update()

            progressbarOne['value'] += 1
            Input_windows.update()
            print("Scanning reference...")
            nirs.scan()
            progressbarOne['value'] += 1
            Input_windows.update()
            results = nirs.get_scan_results()
            wave = results["wavelength"]
            print(type(wave))

            progressbarOne['value'] += 1
            Input_windows.update()
            reference = results["intensity"]

            wavef = open('/home/ispect/Desktop/0302/0309/root_emmc/root_ref_wave.txt', 'w')
            wave = str(wave)
            wave = wave.replace("[", "")
            wave = wave.replace("]", "")
            wavef.write(str(wave))
            wavef.close()
            progressbarOne['value'] += 1
            Input_windows.update()
            wavef = open('/home/ispect/Desktop/0302/0309/root_emmc/root_ref_wave.txt', 'r')
            q = wavef.read().split(", ")
            print(q, type(q), len(q))

            progressbarOne['value'] += 1
            Input_windows.update()
            intensityf = open('/home/ispect/Desktop/0302/0309/root_emmc/root_ref_intensity.txt', 'w')
            intensity = str(reference)
            intensity = intensity.replace("[", "")
            intensity = intensity.replace("]", "")
            intensityf.write(str(intensity))
            intensityf.close()
            
            progressbarOne['value'] += 1
            Input_windows.update()
            intensityf = open('/home/ispect/Desktop/0302/0309/root_emmc/root_ref_intensity.txt', 'r')
            i = intensityf.read().split(", ")
            print(i, type(i), len(i))

            progressbarOne['value'] += 1
            Input_windows.update()

            tkMessageBox.showinfo('warning', "Renew Measure Reference successfully.")
            Input_windows.destroy()

            choosen()

        print("hello world")

    def judge_event_2():
        print("Shutdown.")
        os.system("shutdown -h now")

    RFUID = open('/home/ispect/Desktop/0302/0309/libs/interface_UID.txt', 'r')
    UID = RFUID.read()
    print(UID, type(UID))

    if UID == "C55CD86B":
        data_name = str("Yu-Shing Zhou")
    elif UID == "451B3749":
        data_name = str("Jungle Chen")
    elif UID == "9536C36B":
        data_name = str("Yu-Ping Chen")
    elif UID == "F585CB6B":
        data_name = str("Ya-Pei Chou")
    elif UID == "251B96B":
        data_name = str("Hsin-I Ho")
    elif UID == "75CDAB28":
        data_name = str("Jen-Jie Chieh")
    else:
        data_name = str("NO data")

    Input_windows = tk.Tk()
    Input_windows.title('Enter filename and condition')
    #Input_windows.geometry('1024x600+0+0')
    Input_windows.attributes("-fullscreen", True)
    Input_windows.configure(bg="lightsteelblue")

    img1 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/topic.png')
    img1 = img1.resize((600, 420), Image.ANTIALIAS)
    tk_img_1 = ImageTk.PhotoImage(img1)

    img2 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/test.png')
    img2 = img2.resize((420, 127), Image.ANTIALIAS)
    tk_img_2 = ImageTk.PhotoImage(img2)

    label = tk.Label(Input_windows, image=tk_img_1, width=600, height=420)
    label.config(bg="lightsteelblue")
    label.grid(padx=172, pady=8, sticky='nw')

    UID_check = tk.Label(Input_windows, text="Name = " + data_name + "   UID = " + UID)
    UID_check.config(font=("Times New Roman", 18, "bold"), bg="lightsteelblue")
    UID_check.grid(row=0, column=0)
    UID_check.grid(padx=500, pady=550, sticky='nw')

    Enter_button = tk.Button(Input_windows, image=tk_img_2, command=judge_event)
    Enter_button.grid(row=5, column=1)
    Enter_button.config(bg="white")
    Enter_button.place(x=202, y=427, width=540, height=128)

    progressbarOne = ttk.Progressbar(Input_windows)
    progressbarOne.place(x=272, y=417, width=400, height=10)

    shutdown_button = tk.Button(Input_windows, text='shutdown', command=judge_event_2)
    shutdown_button.grid(row=5, column=1)
    shutdown_button.config(font=("Times New Roman", 9, "bold"), bg="lightsteelblue")
    shutdown_button.place(x=5, y=547, width=100, height=18)

    Input_windows.mainloop()

        
def RFID_CODE():  
    def down():
        print("a")
        os.system("shutdown -h now")
    def measure():
        COM_PORT = '/dev/ttyUSB0'
        BAUD_RATES = 9600
        ser = serial.Serial(COM_PORT, BAUD_RATES)
        data_raw = ser.readline()
        data_raw = data_raw.replace("\r\n", "")
        print('receive original data', data_raw)
        ser.close()

        if data_raw == "C55CD86B":
            data_name = str("Yu-Shing Zhou")
        elif data_raw == "451B3749":
            data_name = str("Jungle Chen")
        elif data_raw == "9536C36B":
            data_name = str("Yu-Ping Chen")
        elif data_raw == "F585CB6B":
            data_name = str("Ya-Pei Chou")
        elif data_raw == "251B96B":
            data_name = str("Hsin-I Ho")
        elif data_raw == "75CDAB28":
            data_name = str("Jen-Jie Chieh")
        else:
            data_name = str("NO data")

        RFUID = open('/home/ispect/Desktop/0302/0309/libs/interface_UID.txt', 'w')
        UID = str(data_raw)
        RFUID.write(UID)
        RFUID.close()

        tkMessageBox.showinfo('warning', "Read UID successfully\nName = " + data_name + "\nUID = " + UID)

        RFID.destroy()
        choosen()

    RFID = tk.Tk()
    RFID.title('Read RFID UID')
    RFID.attributes("-fullscreen", True)
    #RFID.geometry('1024x600+0+0')
    RFID.configure(bg="lightsteelblue")

    img5 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/RFID.png')
    img5 = img5.resize((600, 400), Image.ANTIALIAS)
    tk_img_5 = ImageTk.PhotoImage(img5)

    img4 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/UIDbutton.png')
    img4 = img4.resize((480, 127), Image.ANTIALIAS)
    tk_img_4 = ImageTk.PhotoImage(img4)

    RFIDlabel = tk.Label(RFID, image=tk_img_5, width=600, height=400)
    RFIDlabel.config(bg="lightsteelblue")
    RFIDlabel.grid(padx=172, pady=5, sticky='nw')

    Measure_button = tk.Button(RFID, image=tk_img_4, command=measure)
    Measure_button.grid(row=5, column=1)
    Measure_button.config(bg="white")
    Measure_button.place(x=202, y=437, width=540, height=128)

    shutdown_button = tk.Button(RFID, text='shutdown', command=down)
    shutdown_button.grid(row=5, column=1)
    shutdown_button.config(font=("Times New Roman", 9, "bold"), bg="lightsteelblue")
    shutdown_button.place(x=5, y=547, width=100, height=18)

    RFID.mainloop()

def choosen():
    def execute_2():
        print("QAQ")
        os.system("shutdown -h now")
    def cuppon():
        choos.destroy()
        PCA_detectionWindows()
    def clothes():
        choos.destroy()
        QAQ_detectionWindows()

    RFUID = open('/home/ispect/Desktop/0302/0309/libs/interface_UID.txt', 'r')
    UID = RFUID.read()
    print(UID,type(UID))

    if UID == "C55CD86B":
        data_name = str("Yu-Shing Zhou")
    elif UID == "451B3749":
        data_name = str("Jungle Chen")
    elif UID == "9536C36B":
        data_name = str("Yu-Ping Chen")
    elif UID == "F585CB6B":
        data_name = str("Ya-Pei Chou") 
    elif UID == "251B96B":
        data_name = str("Hsin-I Ho") 
    elif UID == "75CDAB28":
        data_name = str("Jen-Jie Chieh") 
    else:
        data_name = str("NO data")

    choos = tk.Tk()
    choos.title('Read RFID UID')
    choos.attributes("-fullscreen", True)
    #choos.geometry('1024x600+0+0')
    choos.configure(bg = "lightsteelblue")    

    img1 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/cup.png')
    img1 = img1.resize((245,400), Image.ANTIALIAS)
    tk_img_1 = ImageTk.PhotoImage(img1)

    img2 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/clothes.png')
    img2 = img2.resize((250,400), Image.ANTIALIAS)
    tk_img_2 = ImageTk.PhotoImage(img2) 

    img3 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/chosen.png')
    img3 = img3.resize((600,140), Image.ANTIALIAS)
    tk_img_3 = ImageTk.PhotoImage(img3) 

    chooselabel = tk.Label(choos, image=tk_img_3, width=600, height=140)  
    chooselabel.config(bg = "lightsteelblue")
    chooselabel.grid(padx=5, pady=5, sticky='nw')

    UID_check = tk.Label(choos, text="Name = " + data_name + "   UID = " + UID)
    UID_check.config(font=("Times New Roman",18,"bold"),bg = "lightsteelblue" )
    UID_check.grid(row=0, column=0)
    UID_check.grid(padx=500, pady=550, sticky='nw') 

    CUP_button = tk.Button(choos, image=tk_img_1, command=cuppon)
    CUP_button.grid(row=5, column=1)
    CUP_button.config(font=("Times New Roman",18,"bold"),bg = "white")
    CUP_button.place(x = 160, y = 150, width=244, height=400)

    Clothes_button = tk.Button(choos, image=tk_img_2, command=clothes)
    Clothes_button.grid(row=5, column=1)
    Clothes_button.config(font=("Times New Roman",18,"bold"),bg = "white")
    Clothes_button.place(x = 560, y = 150, width=250, height=400)

    shutdown_button = tk.Button(choos, text='shutdown', command=execute_2)
    shutdown_button.grid(row=5, column=1)
    shutdown_button.config(font=("Times New Roman",9,"bold"),bg = "lightsteelblue")
    shutdown_button.place(x = 5, y = 557, width=100, height=18)  

    choos.mainloop()

def OMG():
    def judge_event():
        QQ.destroy()
        RFID_CODE()
    def judge_event_2():
        print("hello world")
        matebdTk2.Engineer_mode()
        RFID_CODE()

    QQ = tk.Tk()
    QQ.title('Enter filename and condition')
    #QQ.geometry('1024x600+0+0')
    QQ.attributes("-fullscreen", True)
    QQ.configure(bg = "lightsteelblue")
    #Input_windows.configure(bg = "lightsteelblue")
    #Input_windows.attributes('-fullscreen', True)

    img1 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/topic.png')
    img1 = img1.resize((600,420), Image.ANTIALIAS)
    tk_img_1 = ImageTk.PhotoImage(img1)   

    img2 = Image.open('/home/ispect/Desktop/0302/0309/GUI_element/usermodel.png')
    img2 = img2.resize((420,127), Image.ANTIALIAS)
    tk_img_2 = ImageTk.PhotoImage(img2)    

    label = tk.Button(QQ, image=tk_img_1, bd=0, relief = "solid", activebackground = "lightsteelblue", highlightthickness = 0, command = judge_event_2)  
    label.config(bg = "lightsteelblue")
    label.place(x=172, y=8, width=600, height=420)

    Enter_button = tk.Button(QQ, image=tk_img_2, command=judge_event)
    Enter_button.grid(row=5, column=1)
    Enter_button.config(bg = "white")
    Enter_button.place(x = 208, y = 427, width=540, height=128)
        
    QQ.mainloop()
