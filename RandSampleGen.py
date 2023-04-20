import random

sentences = [    
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",    
    "All work and no play makes Jack a dull boy.",    
    "Life is like a box of chocolates, you never know what you're gonna get.",    
    "To be or not to be, that is the question.",    
    "All that glitters is not gold.",    
    "The best things in life are free.",    
    "Practice makes perfect.",    
    "Actions speak louder than words.",    
    "Time heals all wounds.",    
    "When in Rome, do as the Romans do.",    
    "An apple a day keeps the doctor away.",    
    "Beauty is in the eye of the beholder.",    
    "Better late than never.",    
    "Birds of a feather flock together.",    
    "Every cloud has a silver lining.",    
    "Honesty is the best policy.",    
    "If at first you don't succeed, try, try again.",    
    "Necessity is the mother of invention.",    
    "The early bird catches the worm."
]

def get_sample(size=80):
    result = []
    for _ in range(size):
        sentence1 = random.choice(sentences)
        sentence2 = random.choice(sentences)
        sentence3 = "I'm acting as a context for them"
        label = random.choice([0, 1, 2])
        
        result.append((sentence1, sentence2, sentence3, label))

    return result
