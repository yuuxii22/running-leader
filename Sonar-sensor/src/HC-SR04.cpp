// Source link:
// https://www.intorobotics.com/how-to-use-sensor_msgs-range-ros-for-multiple-sensors-with-rosserial/

// For Newping.h:
// https://playground.arduino.cc/Code/NewPing/#Constructor. (Arduino website)
// https://github.com/microflo/ NewPing/blob/master/NewPing.h (Source)
#include <NewPing.h>

// Support filter.
#include <SimpleKalmanFilter.h>

// available in ROS docs
#include <ros.h>
#include <ros/time.h>
#include <sensor_msgs/Range.h>

#define SONAR_NUM 1       // The number of sensors.
#define MAX_DISTANCE 200  // Mad distance to detect obstacles.
#define PING_INTERVAL 33  // Looping the pings after 33 microseconds.

unsigned long pingTimer[SONAR_NUM];  // Holds the times when the next ping
                                     // should happen for each sensor.
unsigned int cm[SONAR_NUM];          // Where the ping distances are stored.
uint8_t currentSensor = 0;           // Keeps track of which sensor is active.

unsigned long _timerStart = 0;

size_t LOOPING = 40;  // Loop for every 40 milliseconds.

uint8_t oldSensorReading[SONAR_NUM];  // Store last valid value of the sensors.

uint8_t Sensor;  // Store raw sensor's value.

uint8_t SensorKalman;  // Store filtered sensor's value.

NewPing sonar[SONAR_NUM] = {NewPing(3, 2, MAX_DISTANCE)};

/*
   Template: SimpleKalmanFilter(e_mea, e_est, q);
   e_mea: Measurement Uncertainty
   e_est: Estimation Uncertainty
   q: Process Noise
*/
SimpleKalmanFilter KF(2, 2, 0.01);

ros::NodeHandle nh;  // create an object which represents the ROS node.

// looping the sensors. We only have 1
void sensorCycle() {
  for (uint8_t i = 0; i < SONAR_NUM; i++) {
    if (millis() >= pingTimer[i]) {
      pingTimer[i] += PING_INTERVAL * SONAR_NUM;
      if (i == 0 && currentSensor == SONAR_NUM - 1) oneSensorCycle();
      sonar[currentSensor].timer_stop();
      currentSensor = i;
      cm[currentSensor] = 0;
      sonar[currentSensor].ping_timer(echoCheck);
    }
  }
}

// If ping received, set the sensor distance to array.
void echoCheck() {
  if (sonar[currentSensor].check_timer())
    cm[currentSensor] = sonar[currentSensor].ping_result / US_ROUNDTRIP_CM;
  // US_ROUNDTRIP_CM = 57: time (in uS) for sound wave to travel 2cm, available
  // in "NewPing.h".
}

// Return the last valid value from the sensor.
void oneSensorCycle() { Sensor = returnLastValidRead(0, cm[0]); }

// If sensor value is 0, then return the last stored value different than 0.
int returnLastValidRead(uint8_t sensorArray, uint8_t cm) {
  if (cm != 0) {
    return oldSensorReading[sensorArray] = cm;
  } else {
    return oldSensorReading[sensorArray];
  }
}

// Apply Kalman Filter to sensor reading.
void applyKF() { SensorKalman = KF.updateEstimate(leftSensor); }

// get the starting time.
void startTimer() { _timerStart = millis(); }

// to loop every millisecond.
bool isTimeForLoop(int _mSec) { return (millis() - _timerStart) > _mSec; }

void sensor_msg_init(sensor_msgs::Range &range_name, char *frame_id_name) {
  range_name.radiation_type = sensor_msgs::Range::ULTRASOUND;
  range_name.header.frame_id = frame_id_name;
  range_name.field_of_view = 0.26;
  range_name.min_range = 0.0;
  range_name.max_range = 2.0;
}

// Create instances for range messages.
sensor_msgs::Range range_;

// Create publisher onjects for all sensors
ros::Publisher pub_range("/ultrasound", &range_);

void setup() {
  pingTimer[0] = millis() + 75;
  for (uint8_t i = 1; i < SONAR_NUM; i++)
    pingTimer[i] = pingTimer[i - 1] + PING_INTERVAL;

  nh.initNode();
  nh.advertise(pub_range);

  sensor_msg_init(range_, "/ultrasound_left");
}

void loop() {
  if (isTimeForLoop(LOOPING)) {
    sensorCycle();
    oneSensorCycle();
    applyKF();
    range_.range = SensorKalman;
    range_.header.stamp = nh.now();
    pub_range.publish(&range_);
    startTimer();
  }
  nh.spinOnce();  // Handle ROS events
}