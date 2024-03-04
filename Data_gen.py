import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from Save_load import *

def data_gen():
    data=pd.read_csv("Data_set/xAPI-Edu-Data.csv")

    print(data.columns)

    data["gender"] = data.gender.replace("F",0)
    data["gender"] = data.gender.replace("M",1)

    data["NationalITy"] = data.NationalITy.replace("KW",0)
    data["NationalITy"] = data.NationalITy.replace("Jordan",1)
    data["NationalITy"] = data.NationalITy.replace("Palestine",2)
    data["NationalITy"] = data.NationalITy.replace("Iraq",3)
    data["NationalITy"] = data.NationalITy.replace("lebanon",4)
    data["NationalITy"] = data.NationalITy.replace("Tunis",5)
    data["NationalITy"] = data.NationalITy.replace("SaudiArabia",6)
    data["NationalITy"] = data.NationalITy.replace("Egypt",7)
    data["NationalITy"] = data.NationalITy.replace("Syria",8)
    data["NationalITy"] = data.NationalITy.replace("USA",9)
    data["NationalITy"] = data.NationalITy.replace("Iran",10)
    data["NationalITy"] = data.NationalITy.replace("Lybia",11)
    data["NationalITy"] = data.NationalITy.replace("Morocco",12)
    data["NationalITy"] = data.NationalITy.replace("venzuela",13)


    data["PlaceofBirth"] = data.PlaceofBirth.replace("venzuela",0)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("KuwaIT",1)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("Jordan",2)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("Iraq",3)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("lebanon",4)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("SaudiArabia",5)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("USA",6)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("Palestine",7)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("Egypt",8)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("Tunis",9)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("Iran",10)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("Syria",11)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("Lybia",12)
    data["PlaceofBirth"] = data.PlaceofBirth.replace("Morocco",13)


    data["StageID"] = data.StageID.replace("MiddleSchool",0)
    data["StageID"] = data.StageID.replace("lowerlevel",1)
    data["StageID"] = data.StageID.replace("HighSchool",2)



    data["GradeID"] = data.GradeID.replace("G-02",0)
    data["GradeID"] = data.GradeID.replace("G-08",1)
    data["GradeID"] = data.GradeID.replace("G-07",2)
    data["GradeID"] = data.GradeID.replace("G-04",3)
    data["GradeID"] = data.GradeID.replace("G-06",4)
    data["GradeID"] = data.GradeID.replace("G-11",5)
    data["GradeID"] = data.GradeID.replace("G-12",6)
    data["GradeID"] = data.GradeID.replace("G-09",7)
    data["GradeID"] = data.GradeID.replace("G-10",8)
    data["GradeID"] = data.GradeID.replace("G-05",9)


    data["SectionID"] = data.SectionID.replace("A",0)
    data["SectionID"] = data.SectionID.replace("B",1)
    data["SectionID"] = data.SectionID.replace("C",2)



    data["Topic"] = data.Topic.replace("IT",0)
    data["Topic"] = data.Topic.replace("French",1)
    data["Topic"] = data.Topic.replace("Arabic",2)
    data["Topic"] = data.Topic.replace("Science",3)
    data["Topic"] = data.Topic.replace("English",4)
    data["Topic"] = data.Topic.replace("Biology",5)
    data["Topic"] = data.Topic.replace("Spanish",6)
    data["Topic"] = data.Topic.replace("Chemistry",7)
    data["Topic"] = data.Topic.replace("Geology",8)
    data["Topic"] = data.Topic.replace("Quran",9)
    data["Topic"] = data.Topic.replace("Math",10)
    data["Topic"] = data.Topic.replace("History",11)


    data["Semester"] = data.Semester.replace("F",0)
    data["Semester"] = data.Semester.replace("S",1)


    data["ParentAnsweringSurvey"] = data.ParentAnsweringSurvey.replace("No",0)
    data["ParentAnsweringSurvey"] = data.ParentAnsweringSurvey.replace("Yes",1)


    data["ParentschoolSatisfaction"] = data.ParentschoolSatisfaction.replace("Bad",0)
    data["ParentschoolSatisfaction"] = data.ParentschoolSatisfaction.replace("Good",1)


    data["StudentAbsenceDays"] = data.StudentAbsenceDays.replace("Under-7",0)
    data["StudentAbsenceDays"] = data.StudentAbsenceDays.replace("Above-7",1)


    data["Class"] = data.Class.replace("M",0)
    data["Class"] = data.Class.replace("H",1)
    data["Class"] = data.Class.replace("L",2)


    data.drop('Relation', axis=1, inplace=True)

    features = data.iloc[:,:-1]
    label = data.iloc[:,-1]

    feat_skew = features.skew(axis=1)
    feat_kurtosis = features.kurtosis(axis=1)
    feat_mean = features.mean(axis=1)
    feat_median = features.median(axis=1)
    feat_std = features.std(axis=1)
    feat_variance = features.var(axis=1)

    feat_entropy = entropy(features,axis=1)

    coeff_corr=features.iloc[:, :].corrwith(features.iloc[0, :].astype(float), axis=1)


    features = features.assign(skewness=feat_skew,kurtosis=feat_kurtosis,entropy=feat_entropy,mean=feat_mean,median=feat_median,std=feat_std,variance=feat_variance,coeff_corr=coeff_corr)

    features = np.array(features)
    label = np.array(label)

    X_train, X_test, Y_train, Y_test = train_test_split(features,label, test_size=0.3)

    save('x_train', X_train)
    save('y_train', Y_train)
    save('x_test', X_test)
    save('y_test', Y_test)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
data_gen()
