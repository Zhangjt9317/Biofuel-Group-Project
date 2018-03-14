#template platform
import sys
from tkinter import *
import numpy as np
import tkinter as tk
import matplotlib
import tkinter.messagebox as tkMessageBox
import urllib
import json
import pandas as pd
import numpy as np

#matplotlib settup
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from tkinter import filedialog as tkFilledDialog
from clfdata import *
from core import *
#change the button to ttk

style.use("ggplot")

class QSPR_Modeling(tk.Tk):

    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)
        tk.Tk.wm_title(self,"QSPR Modeling")

        container=tk.Frame(self,width=300,height=200,bg="purple")
        container.pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)

        self.frames={}
        for F in (Model_Selection,Models,Graphs):
            frame = F(container,self)
            self.frames[F]= frame
            frame.grid(row=0,column=0,sticky="nsew") #assign the position
        self.show_frame(Model_Selection)

    def show_frame(self,cont):
        frame=self.frames[cont]
        frame.tkraise()

class Model_Selection(tk.Frame):
    def quit(self):
        msg=tkMessageBox.askyesno("Model_Selection","Are you sure to quit?")
        if (msg):
            exit()

    def ref(self):
        tkMessageBox.showinfo("Reference: ","[1].Saldana, D. A.; Starck, L.; Mougin,center P.; Rousseau, B.; Pidol, L.;"
                                            " Jeuland, N.; Creton, B. Energy & Fuels 2011, 25 (9)")

    def acknow(self):
        tkMessageBox.showinfo("Acknowledgements: ","This software is supported by DIRECT program in University of Washington CEI Group. "
                                                   "Designers include: Jingtian Zhang, Renlong Zheng, Zeng Cheng, Chenggang Xi")

    def intro(self):
        tkMessageBox.showinfo("Introduction:","The main goal of this software was to predict flash points "
                                              "and cetane number of an known biofuel. The user can input CID number"
                                              "and obtain the predicted values gained by machine learning alogrithms built"
                                              "in this software. Classification and linear regression were used to train and test models.")

    def __init__(self,parent,controller):

        #CID should be a global variable in order to go through every step of QSPR modeling
        '''This class configures and populates the toplevel window.
                   top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#d9d9d9'  # X11 color: 'gray85'
        font11 = "-family {Segoe UI Black} -size 12 -weight bold " \
                 "-slant roman -underline 0 -overstrike 0"
        font12 = "-family {Segoe UI Black} -size 11 -weight bold " \
                 "-slant roman -underline 0 -overstrike 0"
        font13 = "-family {Segoe UI Black} -size 11 -weight bold " \
                 "-slant roman -underline 1 -overstrike 0"
        font14 = "-family {Courier New} -size 12 -weight normal -slant" \
                 " roman -underline 0 -overstrike 0"
        font9 = "-family {Segoe UI} -size 9 -weight normal -slant " \
                "roman -underline 0 -overstrike 0"

        tk.Frame.__init__(self, parent)
        container = tk.Frame(self, width=300, height=200, bg="#b197e8")
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        btncancel=Button(self)
        btncancel.place(relx=0.42, rely=0.6, height=73, width=166)
        btncancel.configure(activebackground="#c0c0c0")
        btncancel.configure(activeforeground="#000000")
        btncancel.configure(background="#4f24a6")
        btncancel.configure(disabledforeground="#a3a3a3")
        btncancel.configure(font=font12)
        btncancel.configure(foreground="#ffffff")
        btncancel.configure(highlightbackground="#d9d9d9")
        btncancel.configure(highlightcolor="black")
        btncancel.configure(pady="0")
        btncancel.configure(command=self.quit)
        btncancel.configure(text='''Cancel''')

        btnModelSelection = Button(self)
        btnModelSelection.place(relx=0.6, rely=0.6, height=73, width=170)
        btnModelSelection.configure(activebackground="#d9d9d9")
        btnModelSelection.configure(activeforeground="#000000")
        btnModelSelection.configure(background="#4f24a6")
        btnModelSelection.configure(disabledforeground="#a3a3a3")
        btnModelSelection.configure(font=font12)
        btnModelSelection.configure(foreground="#ffffff")
        btnModelSelection.configure(highlightbackground="#d9d9d9")
        btnModelSelection.configure(highlightcolor="black")
        btnModelSelection.configure(pady="0")
        btnModelSelection.configure(command=lambda:controller.show_frame(Models))
        btnModelSelection.configure(text='''Model Selection''')

        txtCID = Entry(self)
        txtCID.place(x=580, y=200, height=45, width=194)
        txtCID.configure(background="white")
        txtCID.configure(disabledforeground="#a3a3a3")
        txtCID.configure(font=font14)
        txtCID.configure(foreground="#000000")
        txtCID.configure(highlightbackground="#d9d9d9")
        txtCID.configure(highlightcolor="black")
        txtCID.configure(insertbackground="black")
        txtCID.configure(justify=CENTER)
        txtCID.configure(selectbackground="#c4c4c4")
        txtCID.configure(selectforeground="black")
        self.Num = (txtCID.get())

        labelCID = Label(self)
        labelCID.place(relx=0.44, rely=0.41, height=45, width=132)
        labelCID.configure(activebackground="#f9f9f9")
        labelCID.configure(activeforeground="black")
        labelCID.configure(background="#b197e8")
        labelCID.configure(disabledforeground="#a3a3a3")
        labelCID.configure(font=font11)
        labelCID.configure(foreground="#000000")
        labelCID.configure(highlightbackground="#d9d9d9")
        labelCID.configure(highlightcolor="black")
        labelCID.configure(text='''CID''')

        mes59 = Message(self)
        mes59.place(x=410, y=20, height=160, width=536)
        mes59.configure(background="#b197e8")
        mes59.configure(font=font13)
        mes59.configure(foreground="#800080")
        mes59.configure(highlightbackground="#d9d9d9")
        mes59.configure(highlightcolor="black")
        mes59.configure(justify=CENTER)
        mes59.configure(text='''This software is designed to predict flash points of different organics by applying machine learning alogrithms. Linear Regression Methods include: GRNN, OLS, PLS, MLPR, KNN and NLR. Classifiers include: KNN''')
        mes59.configure(width=536)

        btnacknow = Button(self)
        btnacknow.place(relx=0.79, rely=0.70, height=73, width=196)
        btnacknow.configure(activebackground="#d9d9d9")
        btnacknow.configure(activeforeground="#000000")
        btnacknow.configure(background="#4f24a6")
        btnacknow.configure(disabledforeground="#a3a3a3")
        btnacknow.configure(font=font12)
        btnacknow.configure(foreground="#ffffff")
        btnacknow.configure(highlightbackground="#d9d9d9")
        btnacknow.configure(highlightcolor="#ffffff")
        btnacknow.configure(pady="0")
        btnacknow.configure(command=self.acknow)
        btnacknow.configure(text='''Acknowledgements''')

        btnref = Button(self)
        btnref.place(relx=0.42, rely=0.80, height=73, width=166)
        btnref.configure(activebackground="#d9d9d9")
        btnref.configure(activeforeground="#4f24a6")
        btnref.configure(background="#5025a7")
        btnref.configure(disabledforeground="#a3a3a3")
        btnref.configure(font=font12)
        btnref.configure(foreground="#ffffff")
        btnref.configure(highlightbackground="#d9d9d9")
        btnref.configure(highlightcolor="black")
        btnref.configure(pady="0")
        btnref.configure(command=self.ref)
        btnref.configure(text='''Reference''')

        btnIntro = Button(self)
        btnIntro.place(relx=0.6, rely=0.80, height=73, width=166)
        btnIntro.configure(activebackground="#d9d9d9")
        btnIntro.configure(activeforeground="#000000")
        btnIntro.configure(background="#4f24a6")
        btnIntro.configure(disabledforeground="#a3a3a3")
        btnIntro.configure(font=font12)
        btnIntro.configure(foreground="#ffffff")
        btnIntro.configure(highlightbackground="#d9d9d9")
        btnIntro.configure(highlightcolor="black")
        btnIntro.configure(pady="0")
        btnIntro.configure(command=self.intro)
        btnIntro.configure(text='''Introduction''')

        lab40 = Label(self)
        lab40.place(relx=-0.01, rely=0.0, height=406, width=402)
        lab40.configure(activebackground="#f9f9f9")
        lab40.configure(activeforeground="black")
        lab40.configure(background="#452c92")
        lab40.configure(disabledforeground="#a3a3a3")
        lab40.configure(foreground="#000000")
        lab40.configure(highlightbackground="#d9d9d9")
        lab40.configure(highlightcolor="black")
        _img1 = PhotoImage(file="./2.png")
        lab40.configure(image=_img1)
        lab40.configure(text='''Label''')

class Models(Model_Selection):

    def __init__(self,parent,controller):
        '''This class configures and populates the toplevel window.
                   top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#d9d9d9'  # X11 color: 'gray85'
        font10 = "-family {Courier New} -size 10 -weight normal -slant" \
                 " roman -underline 0 -overstrike 0"
        font12 = "-family {Segoe UI Black} -size 11 -weight bold " \
                 "-slant italic -underline 0 -overstrike 0"
        font13 = "-family {Segoe UI} -size 11 -weight normal -slant " \
                 "roman -underline 0 -overstrike 0"

        tk.Frame.__init__(self,parent)
        lab42 = LabelFrame(self)
        lab42.place(x=30, y=20, height=395, width=180)
        lab42.configure(relief=RAISED)
        lab42.configure(font=font12)
        lab42.configure(foreground="#000000")
        lab42.configure(relief=RAISED)
        lab42.configure(text='''Numerical Regression''')
        lab42.configure(background="#c0c0c0")
        lab42.configure(width=180)

        lab42_lis46 = Listbox(lab42, exportselection=0)
        lab42_lis46.place(x=40, y=190, height=178, width=104)
        lab42_lis46.configure(background="white")
        lab42_lis46.configure(disabledforeground="#a3a3a3")
        lab42_lis46.configure(font=font10)
        lab42_lis46.configure(foreground="#000000")
        lab42_lis46.configure(width=104)
        lab42_lis46.insert(0, "GRNN")
        lab42_lis46.insert(1, "OLS")
        lab42_lis46.insert(2, "PLS")
        lab42_lis46.insert(3, "PNR")
        lab42_lis46.insert(4, "MLPR")
        lab42_lis46.bind('<<ListboxSelect>>', self.Curselect2) #linear regressions

        labelModels1 = Label(lab42)
        labelModels1.place(relx=0.22, rely=0.28, height=46, width=105)
        labelModels1.configure(activebackground="#808080")
        labelModels1.configure(background="#c0c0c0")
        labelModels1.configure(disabledforeground="#a3a3a3")
        labelModels1.configure(font=font13)
        labelModels1.configure(foreground="#000000")
        labelModels1.configure(text='''Methods''')
        labelModels1.configure(width=105)

        lab43 = LabelFrame(self)
        lab43.place(x=240, y=20, height=395, width=180)
        lab43.configure(relief=RAISED)
        lab43.configure(font=font12)
        lab43.configure(foreground="black")
        lab43.configure(relief=RAISED)
        lab43.configure(text='''Classification''')
        lab43.configure(background="#c0c0c0")
        lab43.configure(highlightbackground="#f0f0f0f0f0f0")
        lab43.configure(highlightcolor="#c0c0c0")
        lab43.configure(width=180)

        lab43_lis47 = Listbox(lab43, exportselection=0)
        lab43_lis47.place(x=40, y=190, height=178, width=104)
        lab43_lis47.configure(background="white")
        lab43_lis47.configure(disabledforeground="#a3a3a3")
        lab43_lis47.configure(font=font10)
        lab43_lis47.configure(foreground="#000000")
        lab43_lis47.configure(width=104)
        lab43_lis47.insert(0, "KNN")
        lab43_lis47.insert(1, "LDA")
        lab43_lis47.insert(2, "SVM")
        lab43_lis47.bind('<<ListboxSelect>>', self.Curselect1) #classes
        lab43_lis47.configure()

        lab44 = LabelFrame(self)
        lab44.place(x=450, y=20, height=395, width=180)
        lab44.configure(relief=RAISED)
        lab44.configure(font=font12)
        lab44.configure(foreground="black")
        lab44.configure(relief=RAISED)
        lab44.configure(text='''Properties''')
        lab44.configure(background="#c0c0c0")
        lab44.configure(highlightbackground="#f0f0f0f0f0f0")
        lab44.configure(highlightcolor="#c0c0c0")
        lab44.configure(width=180)

        lab42_lis46 = Listbox(lab44, exportselection=0)
        lab42_lis46.place(x=40, y=190, height=178, width=104)
        lab42_lis46.configure(background="white")
        lab42_lis46.configure(disabledforeground="#a3a3a3")
        lab42_lis46.configure(font=font10)
        lab42_lis46.configure(foreground="#000000")
        lab42_lis46.configure(width=104)
        lab42_lis46.insert(0, "CN")
        lab42_lis46.insert(1, "FP")
        lab42_lis46.bind('<<ListboxSelect>>', self.Curselect3) #Properties

        labelModels2 = Label(lab43)
        labelModels2.place(relx=0.22, rely=0.28, height=46, width=105)
        labelModels2.configure(background="#c0c0c0")
        labelModels2.configure(disabledforeground="#a3a3a3")
        labelModels2.configure(font=font13)
        labelModels2.configure(foreground="#000000")
        labelModels2.configure(text='''Methods''')
        labelModels2.configure(width=105)

        labelModels2 = Label(lab44)
        labelModels2.place(relx=0.22, rely=0.28, height=46, width=105)
        labelModels2.configure(background="#c0c0c0")
        labelModels2.configure(disabledforeground="#a3a3a3")
        labelModels2.configure(font=font13)
        labelModels2.configure(foreground="#000000")
        labelModels2.configure(text='''Properties''')
        labelModels2.configure(width=105)

        btnbegin = Button(self)
        btnbegin.place(relx=0.77, rely=0.6, height=63, width=106)
        btnbegin.configure(activebackground="#d9d9d9")
        btnbegin.configure(activeforeground="#000000")
        btnbegin.configure(background="#d9d9d9")
        btnbegin.configure(disabledforeground="#a3a3a3")
        btnbegin.configure(foreground="#000000")
        btnbegin.configure(highlightbackground="#d9d9d9")
        btnbegin.configure(highlightcolor="black")
        btnbegin.configure(pady="0")
        btnbegin.configure(text='''Begin''')
        btnbegin.configure(command=lambda: controller.show_frame(Graphs))
        btnbegin.configure(width=106)

        btnback = Button(self)
        btnback.place(relx=0.77, rely=0.78, height=63, width=106)
        btnback.configure(activebackground="#d9d9d9")
        btnback.configure(activeforeground="#000000")
        btnback.configure(background="#d9d9d9")
        btnback.configure(disabledforeground="#a3a3a3")
        btnback.configure(foreground="#000000")
        btnback.configure(highlightbackground="#d9d9d9")
        btnback.configure(highlightcolor="black")
        btnback.configure(pady="0")
        btnback.configure(text='''Back''')
        btnback.configure(command=lambda: controller.show_frame(Model_Selection))
        btnback.configure(width=106)

    def Curselect1(self, event):
        #Classification
        widget = event.widget
        select = widget.curselection()
        value1 = widget.get(ACTIVE)
        self.value1 = value1
        print("selection1:", select, ": '%s'" % value1)

    def Curselect2(self, event):
        #Linear Regression
        widget = event.widget
        select = widget.curselection()
        value2 = widget.get(ACTIVE)
        self.value2 = value2
        print("selection2:", select, ": '%s'" % value2)

    def Curselect3(self, event):
        #Properties
        widget = event.widget
        select = widget.curselection()
        value3 = widget.get(ACTIVE)
        self.value3 = value3
        print("selection3:", select, ": '%s'" % value3)

    def CID(self):
        self.CID=self.Num
        return self.CID

    def descriptor(self):
        self.X = descriptor_generator(self.CID)
        return self.X

    def family(self):
        if self.value1 == "KNN":
            self.family = predict_family_knn(self.X)
        elif self.value1 == "LDA":
            self.family = predict_family_lda(self.X)
        else:
            self.family = predict_family_svm(self.X)
        return self.family

    def classes(self):
        if self.value1 == "KNN":
            self.fig_class = test_knn(self.family)
        elif self.value1 == "LDA":
            self.fig_class = test_knn(self.family)
        else:
            self.fig_class = test_knn(self.family)
        return self.fig_class

class Graphs(Models):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Graph Page")
        label.pack(pady=10, padx=10)
        button1 = tk.Button(self, text="Back to Model Selection", command=lambda: controller.show_frame(Model_Selection))
        button1.pack()

        f=Figure(figsize=(10,4))
        f.add_subplot(111)
        plot_knn("Esters")

        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        toolbar=NavigationToolbar2TkAgg(canvas,self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP,fill=tk.BOTH,expand=True)


app=QSPR_Modeling()
app.mainloop()