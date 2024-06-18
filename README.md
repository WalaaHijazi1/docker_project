
██████╗░░█████╗░░█████╗░██╗░░██╗███████╗██████╗░  ██████╗░██████╗░░█████╗░░░░░░██╗███████╗░█████╗░████████╗
██╔══██╗██╔══██╗██╔══██╗██║░██╔╝██╔════╝██╔══██╗  ██╔══██╗██╔══██╗██╔══██╗░░░░░██║██╔════╝██╔══██╗╚══██╔══╝
██║░░██║██║░░██║██║░░╚═╝█████═╝░█████╗░░██████╔╝  ██████╔╝██████╔╝██║░░██║░░░░░██║█████╗░░██║░░╚═╝░░░██║░░░
██║░░██║██║░░██║██║░░██╗██╔═██╗░██╔══╝░░██╔══██╗  ██╔═══╝░██╔══██╗██║░░██║██╗░░██║██╔══╝░░██║░░██╗░░░██║░░░
██████╔╝╚█████╔╝╚█████╔╝██║░╚██╗███████╗██║░░██║  ██║░░░░░██║░░██║╚█████╔╝╚█████╔╝███████╗╚█████╔╝░░░██║░░░
╚═════╝░░╚════╝░░╚════╝░╚═╝░░╚═╝╚══════╝╚═╝░░╚═╝  ╚═╝░░░░░╚═╝░░╚═╝░╚════╝░░╚════╝░╚══════╝░╚════╝░░░░╚═╝░░░




█░█░█ █▀▀ █░░ █▀▀ █▀█ █▀▄▀█ █▀▀   ▀█▀ █▀█   █▀▄▀█ █▄█   █▀▄ █▀█ █▀▀ █▄▀ █▀▀ █▀█   █▀█ █▀█ █▀█ ░░█ █▀▀ █▀▀ ▀█▀   ▀ ▀▄
▀▄▀▄▀ ██▄ █▄▄ █▄▄ █▄█ █░▀░█ ██▄   ░█░ █▄█   █░▀░█ ░█░   █▄▀ █▄█ █▄▄ █░█ ██▄ █▀▄   █▀▀ █▀▄ █▄█ █▄█ ██▄ █▄▄ ░█░   ▄ ▄▀


Here I want to explain all the steps that I did in order to accomplish my docker project
Here are steps of how I did the whole project:

1) Launch an EC2 instance in AWS, with UBUNTU AMI, t2.meduim instance type (if you can make it bigger like t2.large or even t3 it would be better),
a key pair with a proper name, in the Network Settings choose your VPC witha public Subnet that enables the auto suggestion IP,
VERY IMPORTANT to increase Configure storage than 8 GiB (I had problems with the storage and I had to delete things in order to make space
for the currernt process).

2) after that open the instance you launched after you made sure it is running in a hosted  hypervisor like VirtualBox or MobaXterm
by running the following:

ssh -i <your-key-pair>.pem ubuntu@<Public-IPv4-address>

3) When I entered the instance I made sure to install the appropriate packages, mainly git and docker 
to install git I ran: sudo apt-get update 
and then I ran: sudo apt install git

then to install docker I followed the instructions in the link: 
https://docs.docker.com/engine/install/ubuntu/

I cloned the git repository of the course:
https://github.com/AlexeyMihaylovDev/atech-devops-nov-2023
(you might not be able to enter it because only students of the course have an access to it)

finally, I moved the yolo5 file from the course directrory to the directroy of the docker_project that I created.

4) In this step, I deployed  a MongoDB cluster (replica set) with docker.
it is a database, offers high availability deployment using multiple replica sets.
A replica set is a group of MongoDB servers, called nodes, containing an identical copy of the data.
If one of the servers fails, the other two will pick up the load while the crashed one restarts, without any data loss.

to deploy the mongo I followed the steps in this link:
https://www.mongodb.com/resources/products/compatibilities/deploying-a-mongodb-cluster-with-docker

It is important to note that MongoDB is a source-available, cross-platform, document-oriented database program. Classified as a NoSQL database product,
to read more about NoSQL please visit the following link:
https://www.mongodb.com/resources/basics/databases/nosql-explained/nosql-vs-sql

5) The yolo5/app.py app is a flask based webserver, with a single endpoint /predict, which can be used to predict objects in images.
To use this endpoint, you don’t send the image directly in the HTTP request. Instead, you attach a query parameter called imgName to the URL (e.g. localhost:8081/predict?imgName=street.jpeg), which represents an image name stored in an S3 bucket.
The service downloads this image from the S3 bucket and detect objects in it.
Take a look on the code, and complete the # TODOs. Feel free to change/add any functionality as you wish!

So, I completed the #TODO tasks in the app.py file in order to make a well functionality code.

6) After completing the #TODO taskes, it is important to check if the app.py in the yolo5 works by running: python3 app.py 
inside the yolo5 file, if the python3 package is not installed then run:
sudo apt-get update
sudo apt install python3

and then install the pip package by running:
sudo apt install python3-pip

When running the app.py, if there were any issues it should be fixed before creating the container,
the issues I faced were:
packages that is not installed
environment variables that should be exported like BUCKET_NAME (it is the name of my bucket in the aws web service that I already created)

NOTE: Yolo5 have a git repository it is important to clone it, you can find it in the link:
https://github.com/ultralytics/yolov5/tree/master

if you visit the link you can find how to install the packages in the requirements.txt file, I accessed the yolov5 that was created
after clonning and there was the requirenments file just run:
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt


6) The yolo5 app can be running only as a Docker container. 
This is because the app depends on many files that don’t exist on your local machine, but do exist in the ultralytics/yolov5 base image.
Take a look at the Dockerfile, it has instructions to build a Docker image for a Python application that uses the YOLOv5
(You Only Look Once) model for object detection.

I built the image and run it by running the following:
sudo docker build -t yolov5-app:latest .
sudo docker run -d -p 8081:8081 --name yolov5-container -v ~/.aws:/root/.aws -e BUCKET_NAME=<my-bucket-name> \
-e AWS_ACCESS_KEY_ID=<my-aws-access-key-id> -e AWS_SECRET_ACCESS_KEY=<my-aws-secret-access-key> \
-e AWS_REGION=<the-region-that-iam-running-ec2-instance-from> yolov5-app:latest

STEPS TO CREATE AWS ACCESS KEY: Entering my IAM user through the AWS Management Console then by clicking on the "Security credentials", 
scrolling down I found the "Access keys" section, then I clicked on creating a key, and I downloaded the csv file that has the data in it.

and then I was able to insert the ACCESS KEY data when I ran the container.

7) Finally, by running:
curl -X POST "localhost:8081/predict?imgName=<name-any-image-in-s3-bucket.jpg"

By running this command I was testing the functionality of the yolo5 application that is running on port 8081 of the local machine
,in my case it is the EC2 instance.

if it worked I was HAPPY
if it didn't work, I ran the command:
sudo docker logs <container-id>
container's id can be obtained by running:
sudo docker ps
OR
sudo docker ps -a
the second one shows the running and the stopped containers.
if the container is in Exited then there must be a problem, so I ran the command:
sudo docker logs <container-id>

The result of all the work till now before and after running the yolo5:

BEFORE:

<img src="https://github.com/WalaaHijazi1/docker_project/assets/151656646/1cc82598-e666-4eec-905d-b400d7a42525.jpg" width="450" height="250">

AFTER:

<img src="https://github.com/WalaaHijazi1/docker_project/assets/151656646/311b0c3f-1ec3-41fd-9332-fc973d9cc264.jpg" width="450" height="250">



              𝙄𝙢𝙥𝙡𝙚𝙢𝙚𝙣𝙩𝙞𝙣𝙜 𝙏𝙝𝙚 𝙋𝙤𝙡𝙮𝙗𝙤𝙩 𝙈𝙞𝙘𝙧𝙤𝙨𝙚𝙧𝙫𝙞𝙘𝙚

In this part I ran the Telegram bot, and started the whole service:

" ABOUT THE TELEGRAMBOT:
The Telegram app is a flask-based service that responsible for providing a chat-based interface for users to interact
with your image object detection functionality.
It utilizes the Telegram Bot API(Application Programing Interface) to receive user 
images and respond with detected objects."


                       ##### TO CREATE A TELEGRAM BOT ######
o       Download and install telegram desktop (you can use your phone app as well).
o       Once installed, create your own Telegram Bot after visiting the link: https://core.telegram.org/bots.
o       You will receive a telegram token for your bot, to use it for later (it should be kept as private).


Implementing bot logic involves running a local Python script that listens for updates from Telegram servers.
When a user sends a message to the bot, Telegram servers forward the message to the Python app using a method
called webhook (long-polling and websocket are other possible methods which wouldn’t be used in this project).
The Python app processes the message, executes the desired logic, and may send a response back to Telegram servers,
 which then delivers the response to the user.
The webhook method consists of simple two steps:

<img src="https://github.com/WalaaHijazi1/docker_project/assets/151656646/8df38995-79e3-445d-bbe1-5671da2e9530.jpg" width="280" height="100">


Setting your chat app URL in Telegram Servers:

Once the webhook URL is set, Telegram servers start sending HTTPS POST requests to the specified webhook URL whenever
there are updates, such as new messages or events, for the bot.

<img src="https://github.com/WalaaHijazi1/docker_project/assets/151656646/9be5aff6-aed9-4309-af4c-f91edd9b0469.jpg" width="280" height="100">


In order for the Telegram servers to work it would need access the webhook URL over the internet in order to send updates,
 a localhost URL need to be set as the webhook for a Telegram bot is a problomatic process.
In order to solve this problem I am using the Ngrok service (https://ngrok.com/) ,
 this service create a secure tunnel between the local machine and a public URL provided by Ngrok.
It exposes the local server to the internet, allowing Telegram servers to reach the webhook URL and send updates to the bot.
To install the Ngrok service, I signed up for the Ngrok service and  followed the instructions in this link:
https://ngrok.com/docs/getting-started/#step-2-install-the-ngrok-agent

Then I Authenticated your ngrok agent by running: 

ngrok config add-authtoken <your-authtoken>

Becaue port 8443 is the port that the telegram bot service will be listening on, I started the ngrok service in the VM by running:

ngrok http 8443


After starting the service I was able to get my bot public URL, it should be like this:

https://16ae-2a06-c701-4501-3a00-ecce-30e9-3e61-3069.ngrok-free.app



                𝙏𝙧𝙮𝙞𝙣𝙜 𝙩𝙝𝙚 𝙗𝙤𝙩 𝙛𝙪𝙣𝙘𝙩𝙞𝙤𝙣𝙖𝙡𝙞𝙩𝙮

After copying the polybot from the course material in git:
Under polybot/bot.py I was given a class called Bot. This class implements a simple telegram bot, as follows:
The constructor __init__ receives the token and telegram_chat_url arguments.
The constructor creates an instance of the TeleBot object, which is a pythonic interface to Telegram API.
I can use this instance to conveniently communicate with the Telegram servers.
Later, the constructor sets the webhook URL to be the telegram_chat_url.
The polybot/app.py is the main app entrypoint. 
It’s nothing but a simple flask webserver that uses a Bot instance to handle incoming messages, caught in the webhook endpoint function.

The default behavior of the Bot class is to “echo” the incoming messages, the results are:

<img src="https://github.com/WalaaHijazi1/docker_project/assets/151656646/3bf7e829-f7a8-46d6-835a-c11ebf5914c0.jpg" width="240" height="140">




                   𝙏𝙝𝙚 𝙌𝙪𝙤𝙩𝙚𝘽𝙤𝙩 𝙘𝙡𝙖𝙨𝙨

In bot.py I was  given a class called QuoteBot which inherits from Bot.
Upon incoming messages, this bot echoing the message while quoting the original message, unless the user is asking politely not to quote.

In app.py, change the instantiated instance to the QuoteBot:

- Bot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)
+ QuoteBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)

Running the QuoteBot I recieved the result:

<img src="https://github.com/WalaaHijazi1/docker_project/assets/151656646/60dd00ee-4ec2-4f77-9e00-4eb25dd9bf71.jpg" width="240" height="140">


                𝘽𝙪𝙞𝙡𝙙𝙞𝙣𝙜 𝙏𝙝𝙚 𝙊𝙗𝙟𝙚𝙘𝙩𝘿𝙚𝙩𝙚𝙘𝙩𝙞𝙤𝙣 𝘾𝙡𝙖𝙨𝙨

in the polybot file there is a bot.py python file, it has an ObjectDetectionBot class with a
handle_message() method that handles incoming messages from end-users.

When users send an image to the bot, first I  have to download the image to the local file system,
then I upload this image to an S3 bucket and perform an HTTP request to the yolo5 service to predict the objects in this image.

In the bot.py python file I comleted the #TODOs in order to make the polybot functions as described above.

Then to check the functionality of the bot, I sent an image through the bot and received the a message of the items in the same
image, and because my S3 bucket in the AWS service I have the same image, so I cfound the result image in the S3 bucket under a prediction
file, the results are:

<<<<<<< HEAD

# My Docker Project

## Project Files and Directories

### polybot
- **Description:** Contains scripts and configurations for the Polybot application.
- **Details:** This directory includes the main logic and setup scripts for running the Polybot.

### yolo5
- **Description:** Includes implementation of the YOLOv5 object detection model.
- **Details:** This folder contains the necessary code and configurations to deploy and run the YOLOv5 model.

### yolov5
- **Description:** This file is a cloned repository from: https://github.com/ultralytics/yolov5 .
- **Details:** It has all the packages that is needed to function the YoloV5 app, and explenation about the AI tool, and it's function.

### .env
- **Description:** Environment variables for the general setup of the project.
- **Details:** This file sets up environment variables used across different parts of the project.

### .env_poly
- **Description:** Specific environment variables for the Polybot application.
- **Details:** Tailored environment settings for running the Polybot application smoothly.

### .env_yolo
- **Description:** Specific environment variables for the YOLO applications.
- **Details:** Custom environment settings for the YOLOv5 implementations.

### docker-compose.yaml
- **Description:** Docker Compose configuration file for orchestrating multiple Docker containers.
- **Details:** Defines the services, networks, and volumes needed for the project’s Docker containers.

### testdisk.log
- **Description:** Log file for disk tests and related diagnostics.
- **Details:** Contains logs and outputs from various disk testing and diagnostics procedures.
=======
Example 1:

<img src="https://github.com/WalaaHijazi1/docker_project/assets/151656646/79aa3ae1-1183-4a59-9ac2-4ac41439c9a6.jpg" width="450" height="400">


Example 2:

<img src="https://github.com/WalaaHijazi1/docker_project/assets/151656646/f0f4ba84-42e7-4619-87ed-2f93b0e1c845.jpg" width="450" height="400">



>>>>>>> origin/main
