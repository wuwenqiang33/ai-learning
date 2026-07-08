# 修改文件名称工具
'''
通过指定文件夹批量修改文件名称
将文件的名称前加上 01 02 03 ...
如果有了序号去掉序号重新排
'''
import os
import TxtFileInfo as fIN

def get_file_list(folder_path):
        file_list = os.listdir(folder_path)
        return file_list

    

if __name__=="__main__":
        # 1.用户输入 指定文件夹
        folder_path = input("请输入指定文件夹路径: ")
        if(folder_path == ""):
                folder_path = r"D:\AI\aiLearning\oneWeek\filePath"

        obj = None
        try:
            obj  = fIN.TxtFileInfo(folder_path)
        except Exception as e:
            print(e)
        finally:
               print("finally")
        if obj is not None:
            print(obj.toString() )
        # 2.获取指定文件夹下的所有文件名称
        file_list = get_file_list(folder_path)
        print(file_list)
        # 3.遍历文件名称列表，打印每个文件名称
        for filename in file_list:
                print(filename)
        new_filenames = []
        new_index_filename = {}
        for index,filename in enumerate(file_list):
                # 判断文件名是否已经有序号，如果有则去掉序号重新排
                if filename[0:2].isdigit() and filename[2] == "_":
                        filename = filename[3:]
                if(index < 9):
                        new_filename = "0"+str(index+1)+"_"+filename
                else:
                        new_filename = str(index+1)+"_"+filename
                new_filenames.append(new_filename)
                new_index_filename[index] = new_filename
        # 4.修改文件名称
        
        print(new_filenames)
        print(new_index_filename)
        for index,filename in enumerate(file_list):
                new_filename = new_index_filename[index]
                os.rename(os.path.join(folder_path,filename),os.path.join(folder_path,new_filename))

                print("修改文件名称成功：%s --> %s"%(filename,new_filename))


