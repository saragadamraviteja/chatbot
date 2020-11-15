#Import the libraries

# from newspaper import Article
import random
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

#Download the punkt package
nltk.download('punkt',quiet=True)

corpus = '''Learning centres are typically a designated area that provides students with exciting and interesting experiences to practice, enrich, reteach, and enhance their learning. At present there are 5 different learning centres in AP and Telangana where our MSIT program is being offered.
MSIT program is being offered by “Consortium of Institutions of Higher Learning” (CIHL), formed by the universities in Andhra Pradesh and Telangana.
You can stay in the guest house in IIIT campus or in one of the 5 hostels, you have to book the room in advance. If the room is available the management will provide you a guest room.
There are two ways to join into MSIT. You can get admitted into MSIT by appearing  and qualifying GAT Exam or GRE score.
The degree will be awarded by the host university in which the student is admitted to.
Consortium of Institutes of Higher Learning is non profit society formed by universities of andhra pradesh, under supervision on APSCHE & Government of Andhra pradesh. CIHL is instrumental in organizing MSIT program in various universities across the state.
MSIT employs a unique ‘Sequential Learning’ technique by which a student only moves on to the next course after thoroughly mastering the preceding course as against the conventional BTECH/MTECH programs. The duration of this program is 2 years. An academic year is divided into 6 mini semesters. Duration of each mini semester is eight weeks. The first year has five IT and Soft Skills mini semesters each and 1 practicum mini semester. The Second year has four IT and Soft Skills mini semesters each and 2 practicum mini semesters.
We offer both IT and  Soft Skills courses in MSIT.
MSIT follows credit based system and information related to acedemic credits will be given at the time of induction
Minimum eligibility criteria to join MSIT  is  having Btech/BE  degree (All branches)
Exam dates may vary. Please check the website for the dates.
We have a medical facility in campus called AAROGYA MEDICAL centre.There is also a visting physiatrist. There is an ambulance in case of emergencies.
Within-campus facilities include a bank extension counter, ATM machine, stationary shop, and laundry services.
Facilities are provided for cricket, volley ball, soccer, Tennis, basket ball. We also have understanding with nearby stadiums for other sports facilities such as badminton, hockey, swimming, etc.
Gymnasium facilities are available in hostel with a physical trainer. Campus has a yoga room with a trained yoga guru for students, faculty and staff.
Yes! The campus is small (66 acres) so the use of cycles itself within the campus is rare to none. The students who have motorized vehicles can park in the parking space near the main gate.
Most of the MSIT mentors are graduates from MSIT having an average teaching experience of 2 years without counting those who have 20 years of experience.
Fee structure may vary from learning center to center.
This is a full time course where you need to attend college regularly. One cannot pursue the course while working.
Several students  obtained PhD after graduating with MSIT.
GAT exam usually takes place in the month of june. Please check this link to know more :https://www.msitprogram.net/admissions/
Learning centres are typically a designated area that provides students with exciting and interesting experiences to practice, enrich, reteach, and enhance their learning. At present there are 5 different learning centres in AP and Telangana where our MSIT program is being offered.
You are only allowed to carry  Hall ticket along with Id proof and you can use a Virtual calcultor in exam.
The management consider only the best marks from both walk in's and GAT, based on that you will be provided ranks, you will receive an email shortly after attending the exam.
The GAT exam is spread over a duration of 150 minutes.
You will be only allowed to take up this exam at particular test centers which you can choose during the application process.
It is a computer based exam.
You can appear for the Walkin exam maximum of three times a year. GAT exam is conducted once a year.
GAT consists of mainly two sections:The Critical Reading section comprises of Reading Comprehension , Analytical and Logical Reasoning or Ability.The Quantitative Ability section comprises of Discrete Comparisons (Percentages, Ratios, Profit and Loss etc.), Data Analysis (Graphs, Charts and Tables), Quantitative Comparison (Diagrams, Formulae, Distance, Time etc.), Sets, Relations and Functions (Venn diagrams, Linear equations, Intersections)
Both GAT and walk-in test are Computer based Entrance Test conducted for the admission into MSIT Program. ... Walk-in entrance test can be taken maximum three times, but a time gap of one week should be there between any two successive attempts. where as GAT exam is conducted once a year.
Minimum requirement to pass each course is 70% or C grade.
Minimum GRE score required to get an admission is 300. GAT cutoff varies yearly and you will be intimated soon after completing the exam.
Hostel facility is available at IIIT hyderabad campus. Staying in hostel is mandatory for the students who are admitted into IIIT. My name is Msit chatbot. I am here to guide you regarding the queries related to MSIT program. MSIT employs a unique ‘Sequential Learning’ technique by which a student only moves on to the next course after thoroughly mastering the preceding course as against the conventional BTECH/MTECH programs which offer courses in a parallel fashion. This sequential pattern would enable a student to master one course a time, understand the relevant concepts, give him some scope to explore in depth into core concepts and move on with the next course. The Mastery Model of assessment requires a student to score 70% and above. This motivates students to stretch their abilities and reach the required target percentage. An academic year is divided into 6 mini semesters. Duration of each mini semester is eight weeks. The first year has five IT and Soft Skills mini semesters each and 1 practicum mini semester. The Second year has four IT and Soft Skills mini semesters each and 2 practicum mini semesters. There would be a 4 - week vacation spread over the year. MSIT offers spectrum of courses including:IT Core,IT Electives,SOFT SKILLS,DOMAIN SPECIALIZATIONS. IT Core:Digital Literacy (Prerequisite),Computational Thinking,Computer Science programming and principles,Introduction to Data science,Object oriented programming,Algorithms and Data Structures - I,Database Management Systems,Software Engineering Foundations,Algorithms and Data Structures - 2,Introduction to computer systems,Computer Networks,Web programming,Mobile programming,Cloud computing,Cyber Security,Practicum - I. IT Electives: Privacy Technologies,Computer Networks. SOFT SKILLS: Communication Skills,Career Management Course,Effective Relationship Building Course,Interview Facing Skills,General aptitude and mental ability & continuous programming assessment. DOMAIN SPECIALIZATIONS:Computer Networks and Information Security,EBusiness Technologies,Software Engineering,Data Analytics & Data Visualization,Machine Learning,Blockchain Technologies. The methodology of “Learning By Doing” is a unique one adopted by MSIT. It aims at giving hands - on experience which helps the students understand the practical implementation aspects and the concepts associated with it. MSIT’s “Learning by Doing” method involves learning in a project-centric way. Learn by Doing helps students to solve problems by applying concepts. Each course is broken into manageable modules each of which is offered as a project to be worked on. Students work in teams with a corporate-like environment. The students understand the concept while working on the project and submit the relevant tasks (deliverables) within the deadline specified by their mentors. The students get the required guidance and support at every level from the mentors who have a wide knowledge base which enhances the knowledge of the students. Each course is offered by a principal mentor and the student to mentor ratio at MSIT stands out to be 10:1 giving enough scope for the students to interact with their mentor whenever required. Personalized mentoring at MSIT aims at nurturing the students in every way. Students are given individual workstations with laptops and 24x7 internet facilities. Rich digital content with strong audio-visual support alongside constant mentoring provides them with the necessary platform, with lectures by eminent people also. Therefore, MSIT students gain hands-on experience from innovative learning solutions.'''

 #Tokenization
text =corpus
sentence_list = nltk.sent_tokenize(text) # a list of sentences

# A function to return a random greeting response to a users greeting

def greeting_response(text):
    text = text.lower()
    #Bots greeting response
    bot_greetings = ['howdy','hi','hello','hola']
    
    #Users greetings
    user_greetings = ['hi','hey','hello','hola','greetings','wassup']
    
    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)

def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0,length))
    
    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                #swap
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index

#create the bots response

def bot_response(user_input):
    user_input = user_input.lower()
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(user_input) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    user_input = ' '.join(filtered_sentence)
    print(user_input)
    sentence_list.append(user_input)
    bot_response = ''
    cm = CountVectorizer().fit_transform(sentence_list)
    similarity_scores = cosine_similarity(cm[-1],cm)
    similarity_scores_list  = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)
    index = index[1:]
    response_flag = 0
    
    j = 0
    for i in range(len(index)):
        print(similarity_scores_list)
        if similarity_scores_list[index[i]] > 0.2:
            bot_response = bot_response+' '+sentence_list[index[i]]
            response_flag = 1
            j = j +1
        if j>2:
            break
        
    if response_flag == 0:
        bot_response = bot_response+' '+"I apologize, I don't understand."
        
    sentence_list.remove(user_input)
        
    return bot_response 
            

#Start the chart
# print('MSIT BOT: I am MSIT ChatBot. I will answer your queries about MSIT Program. If you want to exit, type bye.')
def main(user_input):
    exit_list = ['exit','see you later','bye','quit','break']
    while(True):
        # user_input = input()
        if user_input.lower() in exit_list:
            return 'MSIT BOT: Chat with you later !'
        else:
            if greeting_response(user_input) != None:
                return  greeting_response(user_input)
            else:
                return  bot_response(user_input)