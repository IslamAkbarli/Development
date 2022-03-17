#!/usr/bin/env python
# coding: utf-8

# Load library
import os
import datetime
import numpy as np
import pandas as pd
import win32com.client


# Read excel file
gunluk_yigim = pd.read_excel('Daily_eManat+.xlsx')

# Specify file type and pathes
# DATE = '2022-02-24' # (Year-Month-Day)
FILE_TYPE = ".xlsx"
DATE = pd.to_datetime(gunluk_yigim.Date).max().strftime('%Y-%m-%d')
PATH = (r'C:\Users\huseynovaf\Desktop\Automatic work\eManat plus\files')


# First remove files from folder 
def remove_files(path):
    for files in os.listdir(path):
        os.remove(os.path.join(path, files))


# Download specified files to the folder
def download_files(path, date):
    
    remove_files(path)
    subject = "tarixində ödənişlər - eManat+"
    limit_date = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
    path_download = (r'C:\Users\huseynovaf\Desktop\Automatic work\eManat plus\files' + '\\')
    
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.GetDefaultFolder(6).folders("Günlük Yığımlar Modenis & eManat+")
    messages = inbox.Items

    for message in messages:
        #print('Here for: ' + message.Subject[8:18])
        if (message.Subject.endswith(subject)) & (message.Subject[8:18] > limit_date):
            print('Here for: ' + message.Subject[8:18])
            attachments = message.Attachments
            attachment = attachments.Item(1)
            attachment.SaveASFile(path_download + str(attachment).lower())


# Processing 
def merge_files(merged_file, path, file_type):
    if not os.listdir(path):
        print("Directory is empty")
    else:

        for file in os.listdir(path):
            if file.endswith(file_type):
                print('Found file named:', file)

                daily_report = pd.read_excel(path + '\\' + file)


                current_date = datetime.datetime.strptime(file[:10], '%Y-%m-%d').strftime('%m/%d/%Y')
                print('Date is:', current_date)
                merged_file = pd.concat([merged_file, daily_report], axis = 0).reset_index(drop = True)
                merged_file['Date'].fillna(current_date, inplace = True)
                print(f'{current_date} file is appended!')
                
    #merged_file.Date = merged_file.Date.dt.strftime('%m/%d/%Y').astype(str)
    return merged_file


# Export final processed file and archive
def export_and_archive(file):
    todays_date = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')
    archive_name = "C:\\Users\\huseynovaf\\Desktop\\Automatic work\\eManat plus\\archive\\Daily_eManat+" + todays_date + '.xlsx'
    
    file.to_excel(archive_name, index = False)
    file.to_excel('C:\\Users\\huseynovaf\\Desktop\\Automatic work\\eManat plus\\Daily_eManat+.xlsx', index = False)
    
    print('File is Exported and Archived')



if __name__ == '__main__':
    download_files(PATH, DATE)
    merged_file = merge_files(gunluk_yigim, PATH, FILE_TYPE)
    export_and_archive(merged_file)