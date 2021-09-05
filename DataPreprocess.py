'''
This is util script to modify the dataset filename.
'''


##
import os

##
# for files in os.listdir(r'./data/dailyTest/'):
#     fileYear = files[:4]
#     fileMonth = files[5:7].strip("-")
#     fileDay = files[7:-4].strip("-")
#
#     # print(f"fileYear: {fileYear}")
#     # print(f"fileMonth: {fileMonth}")
#     # print(f"fileDay: {fileDay}")
#     modifyMonth = "0"+fileMonth if len(fileMonth)<2 else fileMonth
#     modifyDay = "0"+fileDay if len(fileDay)<2 else fileDay
#
#     print(f"fileYear: {fileYear}")
#     print(f"fileMonth: {modifyMonth}")
#     print(f"fileDay: {modifyDay}")
#
#     new_file_name = fileYear+modifyMonth+modifyDay+".tif"
#     print("new_file_name:"+new_file_name)
#     os.rename("./data/dailyTest/"+files, "./data/dailyTest/"+new_file_name)

##
for files in sorted(os.listdir(r'./data/dailyTest/')):
    fileYear = files[:4]
    fileMonth = files[4:6]
    fileDay = files[6:-4]

    print(f"fileYear: {fileYear}")
    print(f"fileMonth: {fileMonth}")
    print(f"fileDay: {fileDay}")
