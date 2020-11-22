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
# nltk.download('punkt',quiet=True)

corpus = '''MSIT stands for Masters of science in Information Technology. MSIT full form is Masters of science in Information Technology. 
MSIT program is being offered by “Consortium of Institutions of Higher Learning” (CIHL),is instrumental in organizing MSIT program in various universities across the Andhra Pradesh and Telangana states.
MSIT program is offered in IIIT Hyderabad,JNTU-Hyderabad,JNTU-kakinada,JNTU-Anathapur,SUV universities. 
Learning centres are typically a designated area that provides students with exciting and interesting experiences to practice, enrich, reteach, and enhance their learning. 
At present there are five different learning centres(IIIT Hyderabad,JNTU-Hyderabad,JNTU-kakinada,JNTU-Anathapur,SUV universities)in AP and Telangana where MSIT program is being offered. 
There are two ways to join into MSIT one is through GAT Exam and other is GRE Score. You can get admitted into MSIT by appearing  and qualifying GAT Exam or GRE score. 
The Maters degree will be awarded by the host university in which the student is admitted. 
IIIT campus providing facility for student's parents for staying in Guest house which has to be booked previously. 
Consortium of Institutes of Higher Learning(CIHL) is non profit society formed by universities of Andhra pradesh and Telangana, under supervision on APSCHE & Government of Telangana and Andhra pradesh. 
MSIT employs a unique ‘Sequential Learning’ technique by which a student only moves on to the next course after thoroughly mastering the preceding course as against the conventional BTECH/MTECH programs. 
The duration of MSIT program is two years. The course structure follows: An academic year is divided into 6 mini semesters and the duration of each mini semester is eight weeks. 
The first year has five IT and Soft Skills mini semesters each and one practicum mini semester. The Second year has Specialization and 2 practicum mini semesters. 
MSIT Program offers both IT and  Soft Skills courses. MSIT follows credit based grading system and information related to acedemic credits will be given at the time of induction. 
Minimum eligibility criteria to join MSIT  is  having Minimum Btech/BE  degree (All branches). 
Exam dates may vary so please,check the website for the dates. 
In IIIT Hyderabad, We have a medical facility in campus called AAROGYA MEDICAL centre and there is also a visting physiatrist, an ambulance in case of emergencies. 
Sports Facilities are provided for cricket, volley ball, soccer,badminton, Tennis, basket ball. 
Within-campus facilities include a bank extension counter, ATM banking service facility, stationary shop, and canteens services. 
Gymnasium facilities are available in hostel with a physical trainer and also a yoga room with a trained yoga guru for students, faculty and staff. 
The students who have motorized vehicles can park in the parking space near the main gate. 
Most of the MSIT mentors are graduates from MSIT having an average teaching experience of 2 years without counting those who have 20 years of experience. 
Fee structure may vary from learning center to center. 
This is a full time course where you need to attend college regularly. One cannot pursue the course while working and it is not available for part time. 
Several students  obtained PhD after graduating with MSIT. 
GAT exam usually takes place in the month of june,for more details Please check this link to know more :https://www.msitprogram.net/admissions/ . 
For GAT Exam You are only allowed to carry  Hall ticket along with Id proof and you can use a Virtual calcultor in exam. 
The management consider only the best marks from both walk in's and GAT, based on that you will be provided ranks, you will receive an email shortly after attending the exam. 
GAT Exam is a computer based exam and it is spread over a duration of 150 minutes. 
You can appear for the Walkin exam maximum of three times a year and the GAT exam is conducted once a year. 
GAT Exam pattern consists: The Critical Reading section comprises of Reading Comprehension , Analytical and Logical Reasoning or Ability. 
GAT Exam syllabus: The Quantitative Ability section comprises of Discrete Comparisons (Percentages, Ratios, Profit and Loss etc.), Data Analysis (Graphs, Charts and Tables), Quantitative Comparison (Diagrams, Formulae, Distance, Time etc.), Sets, Relations and Functions (Venn diagrams, Linear equations, Intersections). 
The Minimum requirement to pass in each course is 70% or C grade. 
GAT Exam cutoff varies yearly and you will be intimated soon after completing the exam. 
Hostel facility is available at IIIT Hyderabad campus. 
Staying in hostel is mandatory for the students who are admitted into IIIT Hyderabad. 
In IIIT Hyderabad different Mess facilities available like South,North,Kadamb(Veg),Kadamb(Non-veg),Yukthahar and also three canteens available with in the campus. 
'''

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
    # print(user_input)
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
        # print(similarity_scores_list)
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
    exit_list = ['good bye','exit','see you later','bye','quit','break']
    while(True):
        # user_input = input()
        print(user_input)
        user_input = user_input.lower()
        if user_input in exit_list:
            return 'Bye! chat with you later !'
        else:
            if greeting_response(user_input) != None:
                return  greeting_response(user_input)
            else:
                return  bot_response(user_input)