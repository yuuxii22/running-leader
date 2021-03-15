import tkinter as tk
from time import *
#from geometry_msgs.msg import Twist
#import rospy

MINUTES=0
SECONDS=0
DISTANCE=0
'''
def send_data_to_topic():
    global MINUTES, SECONDS, DISTANCE
    pub = rospy.Publisher('gui_topic', String, queue_size=10)
    rospy.init_node('gui_input', anonymous=True)
    vel_msg = Twist()
    vel_msg.linear.x = MINUTES
    vel_msg.linear.y = SECONDS
    vel_msg.linear.z = DISTANCE
    if not rospy.is_shutdown():
        pub.publish(vel_msg)
'''
def handle_start(event):
    global DISTANCE, MINUTES, SECONDS
    MINUTES = entry_time_minutes.get()
    SECONDS = entry_time_seconds.get()
    DISTANCE = entry_distance.get()  
    print(f"User's target: Done {distance} meters within {MINUTES} minutes, {SECONDS} seconds")
    text_box.insert(tk.END, "minutes: " + str(MINUTES) + "\n")
    text_box.insert(tk.END, "seconds: " + str(SECONDS) + "\n")
    text_box.insert(tk.END, "distance: " + str(DISTANCE) + "\n\n")
    text_box.insert(tk.END, "GET READY!\n")
    send_data_to_topic()

def handle_clear(event):
    global text_box
    entry_time_minutes.delete(0,tk.END)
    entry_time_seconds.delete(0,tk.END)
    entry_distance.delete(0,tk.END)
    text_box.destroy()
    print("User clear target")

if __name__ == '__main__':

    window = tk.Tk()
    window.geometry("1000x200")
    window.title("Running Leader")
    window.columnconfigure([0,1,2], weight=1, minsize=75)
    window.rowconfigure([0,1], weight=1, minsize=50)

    frame_time = tk.Frame(master=window,width=200, height=100,relief=tk.RIDGE)
    label_time = tk.Label(master=frame_time,text='Finishing time (minutes,seconds)')
    entry_time_minutes = tk.Entry(master=frame_time)
    entry_time_seconds = tk.Entry(master=frame_time)
    label_time.pack()
    entry_time_minutes.pack(side=tk.LEFT)
    entry_time_seconds.pack(side=tk.LEFT)
    frame_time.grid(row=0,column=0)

    frame_distance = tk.Frame(master=window,width=200, height=100,relief=tk.RIDGE)
    label_distance = tk.Label(master=frame_distance,text='Insert your target distance (m)')
    entry_distance = tk.Entry(master=frame_distance)
    entry_distance.insert(1,"t")
    label_distance.pack()
    entry_distance.pack()
    frame_distance.grid(row=0,column=2)

    frame_textBox = tk.Frame(master=window,width=5, height=2,relief=tk.RIDGE)
    text_box = tk.Text(master=frame_textBox)
    frame_textBox.grid(row=1,column=0)
    text_box.pack()

    frame_start = tk.Frame(master=window,width=200, height=100,relief=tk.RIDGE)
    start_button = tk.Button(master=frame_start,text="Start",bg="green")
    start_button.bind("<Button-1>",handle_start)
    start_button.pack()
    frame_start.grid(row=1,column=1)

    frame_clear = tk.Frame(master=window,width=200, height=100,relief=tk.RIDGE)
    clear_button = tk.Button(master=frame_clear,text="Clear",bg='yellow')
    clear_button.bind("<Button-1>",handle_clear)
    clear_button.pack()
    frame_clear.grid(row=1,column=2)

    window.mainloop()