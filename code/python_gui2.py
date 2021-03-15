import tkinter as tk
from time import *
#import rospy

TARGET_MINUTES= 0 
TARGET_SECONDS= 0
TARGET_DISTANCE= 0 

#def callback and get_data function

if __name__ == '__main__':
    window = tk.Tk()
    window.geometry("1000x200")
    window.title("Running Leader")
    window.columnconfigure([0,1,2,3,4], weight=1, minsize=75)
    window.rowconfigure([0,1], weight=1, minsize=50)

    frame_time = tk.Frame(master=window,width=200, height=100,relief=tk.RIDGE)
    label_time = tk.Label(master=frame_time,text='Time taken(Minutes Seconds)')
    entry_time_minutes = tk.Entry(master=frame_time)
    entry_time_seconds = tk.Entry(master=frame_time)
    entry_time_minutes.insert(1,'0')
    entry_time_seconds.insert(1,'0')
    label_time.pack()
    entry_time_minutes.pack(side=tk.LEFT)
    entry_time_seconds.pack(side=tk.LEFT)
    frame_time.grid(row=1,column=0)

    frame_distance = tk.Frame(master=window,width=200, height=100,relief=tk.RIDGE)
    label_distance = tk.Label(master=frame_distance,text='Distance travelled')
    entry_distance = tk.Entry(master=frame_distance)
    entry_distance.insert(1,'0')
    label_distance.pack()
    entry_distance.pack()
    frame_distance.grid(row=1,column=2)

    frame_speed = tk.Frame(master=window,width=200, height=100,relief=tk.RIDGE)
    label_speed = tk.Label(master=frame_speed,text='Current speed')
    entry_speed = tk.Entry(master=frame_speed)
    entry_speed.insert(1,'0')
    label_speed.pack()
    entry_speed.pack()
    frame_speed.grid(row=1,column=4)

    frame_target_time = tk.Frame(master=window,width=200, height=100,relief=tk.RIDGE)
    label_target_time= tk.Label(master=frame_target_time,text='Target time(Minutes Seconds)')
    entry_target_time_minutes = tk.Entry(master=frame_target_time)
    entry_target_time_seconds = tk.Entry(master=frame_target_time)
    entry_target_time_minutes.insert(1,str(TARGET_MINUTES))
    entry_target_time_seconds.insert(1,str(TARGET_SECONDS))
    label_target_time.pack()
    entry_target_time_minutes.pack(side=tk.LEFT)
    entry_target_time_seconds.pack(side=tk.LEFT)
    frame_target_time.grid(row=0,column=1)

    frame_target_distance = tk.Frame(master=window,width=200, height=100,relief=tk.RIDGE)
    label_target_distance = tk.Label(master=frame_target_distance,text='Target Distance')
    entry_target_distance = tk.Entry(master=frame_target_distance)
    entry_target_distance.insert(1,str(TARGET_DISTANCE))
    label_target_distance.pack()
    entry_target_distance.pack()
    frame_target_distance.grid(row=0,column=3)

    window.mainloop()