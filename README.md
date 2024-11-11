# Georgia_Albritton_Python_Portfolio
This is Georgia Albritton's portfolio for python code that I learned during BISC 450C. This was my first time to ever code, so everything was very interesting and new to me! I am excited to learn more.


## Analyzing Patient Data 

In this analysis, we looked at inflammation data for multiple patients.
```python
import numpy
```


```python
numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')

```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')

```


```python
print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
print(type(data))
```

    <class 'numpy.ndarray'>



```python
print(data.shape)
```

    (60, 40)



```python
print('first value in data:', data[0,0])
```

    first value in data: 0.0



```python
print('middle value in data:', data[29, 19])
```

    middle value in data: 16.0



```python
print(data[0:4, 0:10])
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(data[5:10, 0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
small = data[:3, 36:]
```


```python
print('small is:')
```

    small is:



```python
print(small)
```

    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
# Lets us a numpy function 
print(numpy.mean(data))
```

    6.14875



```python
maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)
```


```python
print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566



```python
maxval = numpy.amax(data)
minval = numpy.amin(data)
stdval = numpy.std(data)
```


```python
print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566



```python
print('maximum inflammation:', maxval)
print('minimum inflammation:', minval)
print('standard deviation:', stdval)
```

    maximum inflammation: 20.0
    minimum inflammation: 0.0
    standard deviation: 4.613833197118566



```python
# Sometimes we want to look at variation in statistical values, such as maximum inflammation per pateint,
# or average from day one. 

patient_0 = data[0, :] # 0 on first axis (rows), everything on the second (columns)

print('maximum inflammation for patient 0:', numpy.amax(patient_0))
```

    maximum inflammation for patient 0: 18.0



```python
print('maximum inflammation for patient 2:', numpy.amax(data[2, :]))
```

    maximum inflammation for patient 2: 19.0



```python
print(numpy.mean(data, axis = 0))
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]



```python
print(numpy.mean(data, axis = 0).shape)
```

    (40,)



```python
print(numpy.mean(data, axis =1))
```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]


## Using Jupyter Notebooks

```python
%matplotlib inline 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style = "darkgrid")
```


```python
df = pd.read_csv('/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv')
```


```python
df.head()
```


```python
df.tail()
```


```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```


```python
df.head()
```


```python
len(df)
```


```python
df.dtypes
```


```python
non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head()                                         
```


```python
set(df.profit[non_numeric_profits])
```


```python
len(df.profit[non_numeric_profits])
```


```python
bin_sizes, _, _ = plt.hist(df.year[non_numeric_profits], bins= range(1955, 2006) )
```


```python
df = df.loc[~non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```


```python
df.dtypes
```


```python
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x = 0, y = 0)
    
```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit (millions)')
```


```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue (millions)')
```


```python
def plot_with_std(x, y, stds, ax, title, y_label): 
    ax.fill_between(x, y -stds, y + stds, alpha = 0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols= 2)
title = 'Increase in mean and std fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14, 4)
fig.tight_layout()
```


```python

```


```python

```


## Python Fundamentals

```python
# Any python interpreter can be used as a calculator: 
3 + 5 * 4
```




    23




```python
# Lets save a value to a variable
weight_kg = 60
```


```python
print(weight_kg)
```

    60



```python
# Weight0 = valid 
# 0weight = invalid
# weight and Weight are different
```


```python
# Types of data
# There are three common types of data
# Integer numbers 
# floating point numbers
# Strings

```


```python
# Floating point number
weight_kg = 60.3
```


```python
# String comprised of letters
patient_name = "Jon Smith"
```


```python
# String comprised of numbers 
patient_id = '001'
```


```python
# Use variables in p ython 

weight_lb = 2.2 * weight_kg 

print(weight_lb)
```

    132.66



```python
# Lets add a prefix to our patient id 

patient_id = 'inflam_' + patient_id

print(patient_id)
```

    inflam_001



```python
# Lets combine print statements 

print(patient_id, 'weight in kilograms:', weight_kg)
```

    inflam_001 weight in kilograms: 60.3



```python
# we can call a function inside another function 

print(type(60.3))

print(type(patient_id))
```

    <class 'float'>
    <class 'str'>



```python
# We can also do calculations inside the print function 

print ('weight in lbs:', 2.2 * weight_kg)
```

    weight in lbs: 132.66



```python
print(weight_kg)
```

    60.3



```python
weight_kg = 65.0
print('weight in kilograms is now:', weight_kg)
```

    weight in kilograms is now: 65.0

## Storing Values in Lists

```python
odds = [1, 3, 5, 7]
print('odds are:', odds)
```

    odds are: [1, 3, 5, 7]



```python
print('first element:', odds[0])
print('last element:', odds [3])
print(' "-1" element:', odds [-1])
```

    first element: 1
    last element: 7
     "-1" element: 7



```python
names = ['Curie', 'Darwig', 'Turing'] # Typo in Darwin's name

print('names is originally:', names)

names[1] = 'Darwin' # Correct the name 

print('final value of names:', names)
```

    names is originally: ['Curie', 'Darwig', 'Turing']
    final value of names: ['Curie', 'Darwin', 'Turing']



```python
#name = 'Darwin'
#name[0] = 'd'
```


```python
odds.append(11)
print('odds after adding a value:', odds)
```

    odds after adding a value: [1, 3, 5, 7, 11]



```python
removed_element = odds.pop(0)
print('odds after removing the first element:', odds)
print('removed_element:', removed_element)
```

    odds after removing the first element: [3, 5, 7, 11]
    removed_element: 1



```python
odds.reverse()
print('odds after reversing:', odds)
```

    odds after reversing: [11, 7, 5, 3]



```python
odds = [3,5,7]
primes = odds
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7, 2]



```python
odds = [3,5,7]
primes = list(odds)
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7]



```python
binomial_name = "Drosophila melanogaster"
group = binomial_name[0:10]
print('group:', group)

species = binomial_name[11:23]
print('species:', species)

chromosomes = ['X', 'Y', '2', '3', '4']
autosomes = chromosomes[2:5]
print('autosomes:', autosomes)

last = chromosomes[-1]
print('last:', last)
```

    group: Drosophila
    species: melanogaster
    autosomes: ['2', '3', '4']
    last: 4



```python
date = 'Monday 4 January 2023'
day = date[0:6]
print('Using 0 to begin range:', day)
day = date[:6]
print('Omitting beginning index:', day)
```

    Using 0 to begin range: Monday
    Omitting beginning index: Monday



```python
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
sond = months[8:12]
print('With known last position:', sond)

sond = months[8:len(months)]
print('Using len() to get last entry:', sond)

sond = months[8:]
print('Omitting ending index', sond)
```

    With known last position: ['sep', 'oct', 'nov', 'dec']
    Using len() to get last entry: ['sep', 'oct', 'nov', 'dec']
    Omitting ending index ['sep', 'oct', 'nov', 'dec']

## Using Loops

```python
odds = [1, 3, 5, 7]
```


```python
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])

```

    1
    3
    5
    7



```python
odds = [1,3,5]
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-3-01ba67d8a9e5> in <module>
          3 print(odds[1])
          4 print(odds[2])
    ----> 5 print(odds[3])
    

    IndexError: list index out of range



```python
odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

for num in odds: 
    print(num)
```

    1
    3
    5
    7
    9
    11
    13
    15
    17
    19



```python
length = 0
names = ['Curie', 'Darwin', 'Turing']
for value in names:
    length = length + 1
print('There are', length, 'names in the list.')
```

    There are 3 names in the list.



```python
names = "Rosalind"
for name in ['Curie', 'Darwin', 'Turing']:
    print(name)
print('after the loop, name is', name)
```

    Curie
    Darwin
    Turing
    after the loop, name is Turing



```python
print(len([0,1,2,3]))
```

    4



```python
name = ['Curie', 'Darwin', 'Turing']

print(len(name))
```

    3




## Using Multiple Files

```python
import glob
```


```python
print(glob.glob('inflammation*.csv'))
```

    ['inflammation-10.csv', 'inflammation-09.csv', 'inflammation-11.csv', 'inflammation-06.csv', 'inflammation-05.csv', 'inflammation-08.csv', 'inflammation-01.csv', 'inflammation-07.csv', 'inflammation-04.csv', 'inflammation-03.csv', 'inflammation-02.csv', 'inflammation-12.csv']



```python
import glob
import numpy
import matplotlib.pyplot

filenames = sorted(glob.glob('inflammation*.csv'))
filenames = filenames[0:3]

for filename in filenames:
    print(filename)
    
    data = numpy.loadtxt(fname=filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))
    
    axes1 = fig.add_subplot(1,3,1)
    axes2 = fig.add_subplot(1,3,2)
    axes3 = fig.add_subplot(1,3,3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis = 0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis =0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```

    inflammation-01.csv



    <Figure size 1000x300 with 3 Axes>


    inflammation-02.csv



    <Figure size 1000x300 with 3 Axes>


    inflammation-03.csv



    <Figure size 1000x300 with 3 Axes>



##  Making Choices

```python
import numpy
```


```python
data = numpy.loadtxt(fname='inflammation-01.csv', delimiter=',')
```


```python
max_inflammation_0 = numpy.amax(data, axis=0)[0]
```


```python
max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Saspicitious looking maxima!')
```

    Saspicitious looking maxima!



```python
max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Saspicitious looking maxima!')
    
elif numpy.sum(numpy.amin(data, axis=0)) ==0:
    print('Minima add up to zero1')
    
else:
    print('Seems OK!')
```

    Saspicitious looking maxima!



```python
data = numpy.loadtxt(fname = 'inflammation-03.csv', delimiter=',')

max_inflammation_0 = numpy.amax(data, axis = 0)[0]

max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
elif numpy.sum(numpy.amin(data, axis=0)) ==0:
    print('Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!')
else:
    print('Seems ok!')
```

    Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!

## Making Choices

```python
num = 37
if num > 100:
    print('greater')
else: 
    print('not greater')
print('done')
```

    not greater
    done



```python
num = 53
print('before conditional...')
if num > 100:
    print(num, 'is greater than 100')
print('...after conditional')
```

    before conditional...



```python
num = 14

if num > 0: 
    print(num, 'is positive')
elif num == 0:
    print(num, 'is zero')
else:
    print(num, 'is negative')

```

    14 is positive



```python
if (1 > 0) and (1 >= 0):
    print('both parts are true')
else:
    print('at least one part if false')
```

    both parts are true



```python
if (-1 > 0) or (-1 >= 0):
    print('at least one part is true')
else:
    print('both of these are false')
```

    both of these are false



```python
import numpy
```

## Functions

```python
import numpy 
import matplotlib
import matplotlib.pyplot
import glob
```


```python
'freezing point of water in Kelvin:'
def visualize(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize=(10.0, 3.0))
    
    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis=0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis = 0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```


```python
def detect_problems(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    if numpy.amax(data, axis = 0)[0] == 0 and numpy.amax(data, axis=0)[20] == 20:
        print("Suspicious looking maxima!")
    elif numpy.sum(numpy.amin(data, axis=0)) == 0:
        print('Minima add up to zero!')
    else:
        print('Seems ok!')
```


```python
filenames = sorted(glob.glob('inflammation*.csv'))

for filename in filenames:
    print(filename)
    visualize(filename)
    detect_problems(filename)
```


```python
def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value
```


```python
z = numpy.zeros((2,2))
print(offset_mean(z, 3))

```


```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')

print (offset_mean(data, 0))
```


```python
print('orignal min, mean and max are:', numpy.amin(data), numpy.mean(data), numpy.amax(data))
offset_data = offset_mean(data, 0)
print('min, mean, and max of offset data are:',
      numpy.amin(offset_data),
      numpy.mean(offset_data),
      numpy.amax(offset_data))
        
```


```python
print('std dev before and after:', numpy.std(data), numpy.std(offset_data))
```


```python
print('difference in standard deviation before and after:',
      numpy.std(data) - numpy.std(offset_data))
```


```python
# offset_mean(data, target_mean_value):
# return a new array containing the original data with its mean offset to match the desired value. 

def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value
```


```python
def offset_mean(data, target_mean_value):
    """"Return a new array containing the original data with its mean offset to match the desired value"""
    return(data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```


```python
def offset_mean(data, target_mean_value):
    """"Return a new array containing the original data
    with its mean offset to match the desired value.
    
    Examples
    ---------
    
    >>> Offset_mean([1,2,3], 0)
    array([-1., 0., 1.])
    """
    return (data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```


```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')
```


```python
numpy.loadtxt('inflammation-01.csv', ',')
```


```python
def offset_mean(data, target_mean_value = 0.0):
    """"Return a new array containing the original data
    with its mean offset to match the desired value, (0 by default).
    
    Examples
    ---------
    
    >>> offset_mean([1,2,3])
    array([-1., 0., 1.])
    """
    return (data - numpy.mean(data)) + target_mean_value
```


```python
test_data = numpy.zeros((2,2))
print(offset_mean(test_data, 3))
```


```python
print(offset_mean(test_data))
```


```python
def display(a=1, b=2, c=3):
    print('a:', a, 'b', b, 'c:', c)
    
print('no parameters:')
display()
print('one parameter:')
display(55)
print('two parameters:')
display(55,66)
```


```python
print('only setting the value of c')
display(c = 77)
```


```python
help(numpy.loadtxt)
```


```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')
```


```python
def s(p):
    a = 0
    for v in p:
        a += v
    m = a / len(p)
    d = 0
    for v in p:
        d += (v - m) * (v - m)
    return numpy.sqrt(d / (len(p) -1))

def std_dev(sample):
    sample_sum = 0
    for value in sample:
        sample_sum += value
        
    sample_mean = sample_sum / len(sample)
    
    sum_squared_devs = 0
    for value in sample: 
        sum_squared_devs += (value - sample_mean) * (value - sample_mean)
        
    return numpy.sqrt(sum_squared_devs / (len(sample) - 1))

```
## Errors

```python
# This code has an intentional error. You can type it directly
# or use it for reference to understand the error message below. 

def favorite_ice_cream():
    ice_creams = [
        'chacolate',
        'vanilla',
        'strawberry'
    ]
    print(ice_creams[3])
    
favorite_ice_cream()
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-1-96884ae5987c> in <module>
         10     print(ice_creams[3])
         11 
    ---> 12 favorite_ice_cream()
    

    <ipython-input-1-96884ae5987c> in favorite_ice_cream()
          8         'strawberry'
          9     ]
    ---> 10     print(ice_creams[3])
         11 
         12 favorite_ice_cream()


    IndexError: list index out of range



```python
def some_function():
    msg = 'hello, world!'
    print(msg)
    return msg
```


```python
print(a)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-bca0e2660b9f> in <module>
    ----> 1 print(a)
    

    NameError: name 'a' is not defined



```python
print('hello')
```

    hello



```python
count = 0

for number in range(10):
    count = count + number 
print('The count is:', count)
```

    The count is: 45



```python
letters = ['a', 'b', 'c']

print('Letter #1 is', letters[0])
print('Letter #2 is', letters[1])
print('Letter #3 is', letters[2])
#print('Letter #4 is', letters[3])
```

    Letter #1 is a
    Letter #2 is b
    Letter #3 is c



```python
file_handle = open('myfile.txt', 'w')

```
## Defensive Programming 

```python
numbers = [1.5, 2.3, 0.7, -0.001, 4.4]
total = 0.0
for num in numbers: 
    assert num > 0.0, 'Data should only contain positive values'
    total += num 
print('total is:', total)
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-1-9f81f9e6a6b6> in <module>
          2 total = 0.0
          3 for num in numbers:
    ----> 4     assert num > 0.0, 'Data should only contain positive values'
          5     total += num
          6 print('total is:', total)


    AssertionError: Data should only contain positive values



```python
def normalize_rectangle(rect):
    """Normalizes a rectangle so that it is at the origin and 1.0 units long on its longest axis.
    input should be of the format (x0, y0, x1, y1).
    (x0, y0) and (x1, y1) define the lower left and upper right corners of the rectangle respectively."""
    assert len(rect) ==4, 'Rectangles must contain 4 coordinates'
    x0, y0, x1, y1 = rect
    assert x0 < x1, 'Invalid X coordinates'
    assert y0 < y1, 'Invalid Y coordinates'
    
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy: 
        scaled = dy / dx
        upper_x, upper_y = 1.0, scaled
    else:
        scaled = dx / dy
        upper_x, upper_y = scaled, 1.0
        
    assert 0 < upper_x <= 1.0, 'Calculated upper x coordiante invalid'
    assert 0 < upper_y <= 1.0, 'Calculated upper y coordiante invalid'
    
    return (0, 0, upper_x, upper_y)
```


```python
print(normalize_rectangle ( (0.0, 1.0, 2.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-3-266c6a90b65b> in <module>
    ----> 1 print(normalize_rectangle ( (0.0, 1.0, 2.0)))
    

    <ipython-input-2-341ba6237d71> in normalize_rectangle(rect)
          3     input should be of the format (x0, y0, x1, y1).
          4     (x0, y0) and (x1, y1) define the lower left and upper right corners of the rectangle respectively."""
    ----> 5     assert len(rect) ==4, 'Rectangles must contain 4 coordinates'
          6     x0, y0, x1, y1 = rect
          7     assert x0 < x1, 'Invalid X coordinates'


    AssertionError: Rectangles must contain 4 coordinates



```python
print(normalize_rectangle( (4.0, 2.0, 1.0, 5.0) ))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-4-f7e0d48bdfd0> in <module>
    ----> 1 print(normalize_rectangle( (4.0, 2.0, 1.0, 5.0) ))
    

    <ipython-input-2-341ba6237d71> in normalize_rectangle(rect)
          5     assert len(rect) ==4, 'Rectangles must contain 4 coordinates'
          6     x0, y0, x1, y1 = rect
    ----> 7     assert x0 < x1, 'Invalid X coordinates'
          8     assert y0 < y1, 'Invalid Y coordinates'
          9 


    AssertionError: Invalid X coordinates



```python
print(normalize_rectangle( (0.0, 0.0, 1.0, 5.0)))
```

    (0, 0, 0.2, 1.0)



```python
print(normalize_rectangle( (0.0, 0.0, 5.0, 1.0)))
```

    (0, 0, 1.0, 0.2)


## Transcribing DNA into RNA

```python
# Prompt the user to enter the input fasta file name 

input_file_name = input("Enter the name of the input fasta file: ")
```


```python
# Open the input fasta file and read the DNA sequence 

with open(input_file_name, "r") as input_file: 
    dna_sequence = ""
    for line in input_file:
        if line.startswith(">"):
            continue
        dna_sequence += line.strip()
```


```python
# Transcribe the DNA to RNA
rna_sequence = ""
for nucleotide in dna_sequence:
    if nucleotide == "T":
        rna_sequence += "U"
    else:
        rna_sequence += nucleotide
```


```python
# Prompt the user to enter the output file name 

out_file_name = input("Enter the name of the output file: ")
```


```python
# Save the RNA sequence to a text file

with open(out_file_name, "w") as output_file:
    output_file.write(rna_sequence)
    print(f"The RNA sequence has been saved to {out_file_name}")
```


```python
print(rna_sequence)
```

## Translating RNA into Protein

## Coronavirus Genomics
```python

```















