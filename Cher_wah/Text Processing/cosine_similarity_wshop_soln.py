import nltk
import numpy as np
import string
import pandas as pd
import math
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
	"At 20 years of age the will reigns; at 30, the wit; and at 40, the judgement.",
	"Challenges are what make life interesting and overcoming them is what makes life meaningful.",
	"Let your life be shaped by decisions you made, not by the ones you didn't.",
	"Men are not disturbed by things, but the view they take of things.",
	"The privilege of a lifetime is being who you are.",
	"To see the world, things dangerous to come to. To see behind walls, to draw closer. To find each other and to feel. That, is the purpose of life.",
	"We live in deeds, not years; in thoughts, not figures on a dial. We should count time by heart throbs. He most lives who thinks most, feels the noblest, acts the best.",
	"Continuous effort - not strength or intelligence, is the key to unlocking our potential.",
	"Knowledge is knowing what to say. Wisdom is knowing when to say it.",
	"Critique to sharpen; not to put down.",
	"Worry is a product of the future you can't guarantee and guilt is a product of the past we can't change.",
	"Cowards die many times before their deaths; the braves only but once.",
	"Strength doesn't come from what you can do. It comes from overcoming the things you once thought you couldn't.",
	"Our greatest fear should not be of failure, but of succeeding at things in life that don't really matter.",
	"Creativity comes from constraint.",
	"To love someone means you have a desire to change together with that person.",
	"Simple, not Simplistic!",
	"Aspire for the heights but prepare for the depths.",
	"If you would stand well with a great mind, leave him with a favourable impression of yourself; if with a little mind, leave him with a favourable opinion of himself.",
	"What hurts more, the pain of hard work or the pain of regret?",
	"A wise man will make more opportunities than he finds.",
	"Those who have achieved all their aims probably set them too low.",
	"What we fight with is so small, and when we win, it makes us small. What we want is to be defeated, decisively, by successively greater things.",
	"Life is the sum of our choices.",
	"To talk well and eloquently is a very great art, but it is an equally great one to know the right time to stop.",
	"You can live your whole life and never know who you are; until you see the world through the eyes of others.",
	"CHANGE begins, when you start trying.",
	"Design influences meaning.",
	"Nobody made a greater mistake than he who did nothing because he could only do a little.",
	"Many succeed momentarily by what they know; some succeed temporarily by what they do; few succeed permanently by what they are.",
	"You have enemies? Good. That means you've stood up for something, sometime in your life.",
	"When you find peace within yourself, you become the kind of person who can live at peace with others.",
	"Determine never to be idle. No person will have occasion to complain of the want of time, who never loses any. It is wonderful how much may be done, if we are always doing.",
	"A clever person solves a problem; a wise person avoids it.",
	"When I am working on a problem, I never think about beauty. I think only of how to solve the problem. But when I have finished, if the solution is not beautiful, I know it is wrong.",
	"One day you will wake up and there won't be any more time to do the things you have always wanted. Do it now.",
	"If it is important to you, you will find a way. If not, you will find an excuse.",
	"In science, we reserve our highest honors for those who prove we were wrong.",
	"Logic will get you from A to B. Imagination will take you everywhere.",
	"Out of your vulnerabilities will come your strength.",
	"The eyes of others our prisons, their thoughts our cages.",
	"People are attracted to you by what they see in you; they remain attracted to you by what you see in yourself.",
	"Don't be afraid your life will end; be afraid that it will never begin.",
	"Climb the mountain not to plant your flag, but to embrace the challenge, enjoy the air and behold the view. Climb it so you can see the world, not so the world can see you.",
	"The mark of a successful man is one that has spent an entire day on the bank of a river without feeling guilty about it.",
	"In the beginner's mind, there are many possibilities; in the expert's mind, there are few.",
	"I firmly believe that any man's finest hour, the greatest fulfillment of all that he holds dear, is that moment when he has worked his heart out in a good cause and lies exhausted on the field of battle - victorious",
	"The idea of education was to learn to think for yourself.",
	"It is our choices that show what we really are, far more than our abilities.",
	"Before success comes the courage to fail.",
	"A man is rich in proportion to the number of things he can afford to let alone.",
	"Every time we open our mouths, men look into our minds.",
	"A man cannot be comfortable without his own approval.",
	"There are two things to aim at in life: first, to get what you want; and, after that, to enjoy it. Only the wisest of mankind achieve the second.",
	"We all make decisions, but in the end, our decisions made us.",
	"He who establishes his argument by noise and command shows that his reason is weak.",
	"You attract into your life that which you are.",
	"Greatness lies not in being strong, but in the right use of strength.",
	"Diplomacy is the art of telling plain truths without giving offence.",
	"The highest reward for a person's work is not what he gets for it, but what he becomes by it."
]

stop_words = stopwords.words('english') 

porter = nltk.stem.PorterStemmer()

docs = []
punc = str.maketrans('','', string.punctuation)
for doc in corpus:
    doc_no_punc = doc.translate(punc)
    words_stemmed = [porter.stem(w) for w in doc_no_punc.lower().split()
    	if not w in stop_words]
    docs += [' '.join(words_stemmed)]

print(docs)

tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(docs)
tfidf_wm = tfidf_vec.transform(docs).toarray()

features = tfidf_vec.get_feature_names()
indexes = ['doc'+str(i) for i in range(len(corpus))]
tfidf_df = pd.DataFrame(data=tfidf_wm, index=indexes, columns=features)
print(tfidf_df)

query = 'life wise choices'
query = query.translate(punc)	# remove punctuation
query_arr = [' '.join([porter.stem(w) for w in query.lower().split()])]

tfidf_wm2 = tfidf_vec.transform(query_arr).toarray()
print(tfidf_wm2)

print("")
docs_similarity = cosine_similarity(tfidf_wm2, tfidf_df)
query_similarity = docs_similarity[0]

series = pd.Series(query_similarity, index=tfidf_df.index)
sorted_series = series.sort_values(ascending=False)
sorted_series = sorted_series[sorted_series!=0]
print(sorted_series)

print("\nSearch results for query: '", query, "':\n", sep='')

for index in sorted_series.index:
	doc_idx = int(index[3:])
	print(corpus[doc_idx], " [score = ", sorted_series[index], "]\n", sep='')
