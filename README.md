# Covid-19-saves-distance-detection


An AI tool to prevent spreading of coronavirus (COVID-19) by using computer vision on video surveillance.
A social distancing analyzer AI tool to regulate social distancing protocol using video surveillance of CCTV cameras and drones. Social Distancing Analyser to prevent COVID19


## For education purpose only, meant as my contribution towards society

Social Distancing Analyser automatically detects the extent to which social distancing protocols are followed in the area.
Deploying it on current surveillance systems and drones used by police to monitor large areas can help to prevent coronavirus by allowing automated and better tracking of activities happening in the area. It shows analytics of the area in real time. It can also be used to alert police in case of considerable violation of social distancing protocols in a particular area. 

  ### Please fork the repository and give credits if you use any part of it. :slightly_smiling_face:
  ## It took me time and effort to code it.
  ## I would really appreciate if you give it a star. :star:
  ## ASK FOR MY PERMISSION via email Ahmed mohy (a.mohy881@gmail.com) Ahmed omar(ahmedomar.bfci@gmail.com) before publicizing any part of code or output generated from it. Read Licence to avoid problems
  ## Supervised by Prof.: Hala Zayed (hala.zayed@fci.bu.edu.eg) &&  Dr.: Mostafa AbdSalam (mustafa.abdo@ymail.com)                                                  
 



## Features:
* Get the areal time analytics such as:
   - Draw red boxes around near objects 
   - Draw green Boxes around object with save distance 
   - Number of people in a particular area
   - The extent of risk to a particular person.
* Stores a video output for review

## Things needed to be improved :
* Faster processing
#### Please Note: angle factor is needed to be set between 0 to 1 for v2.0 according to the angle of camera (move towards one as angle becomes verticle)
## Installation:
* Fork the repository and download the code.
* Download the following files and place it in the same directory
   - https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
   - https://pjreddie.com/media/files/yolov3.weights
* For slower CPUs, use yolov3-tiny (link in the code comments)
* for yolov3-tiny Download from 
   - https://pjreddie.com/media/files/yolov3-tiny.weights
   - https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
* Install all the dependenices
* Run social_distancing_analyser.py or social_distancing_analyser 2.0.py



copyright © 2020 BFCAI | All rights reserved
