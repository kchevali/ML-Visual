
"""
============================================================
See Trello: https://trello.com/b/h3OzsIph/teachapp-python
See LucidChart:https://lucid.app/lucidchart/be7b3673-68dc-4245-a12f-ce1caa733f3f/edit?beaconFlowId=247D6DF1EC4CD06A&page=0_0#
============================================================
SYMBOLS: ✓☐☒
TODO 8/5
✓Animal Example Online: https://archive.ics.uci.edu/ml/datasets/Zoo
✓Adult Example Online: https://archive.ics.uci.edu/ml/datasets/Adult
✓Medical Example Online: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

TODO 8/12
✓Size of room for model should be a ratio of number of dots
✓Space out dots
✓Give names to files (dont show .csv)
✓Space Partition
    Color room label
    Color class nodes and class button (or add label)
☒TASKS: Works with tasks/steps (Guide user through the app)
    1. What is Decision Tree? -Advantage vs Disadvantage
    2. Example (Describe how to select features and create the tree) - Tutorial
    3. Improvement(Bagging) -
    4. Practice (show different difficulties) - show code (training[70%] vs testing data[30%])
    5. Extended Reading (Teaching materials) - pdf link
☐DATA
    Find examples that have Yes or No labels
☐TREE: Visualize tree

TODO 8/16
✓More robust method of decending view hiearchy (search by id/name/type)
✓Remove left-right/top-bottom button -> use mouse to navigate space partition
✓Allow decision tree partition to multiple children
✓Add scroll to table to show more rows
☒Use text box/audio to help guide the user
☐Select room and highlight
☐Add link to show excel file
☐Allow user to write code (also show library code / manual code [DIY])
☐User interaction / feedback
    (good partition / partition is done / need more partition )
    Give hints

TODO 8/20
✓Change True/False to Yes/No (Liked/Disliked)
✓Change Target Column (#->String)
✓Remove File List for Examples (Show just one example)
✓Show accuracy with the testing data
☒Prediction Improvement - Let the user run the Bagging Algorithm
    -Use larger data set
    -Hyperparameter(Improvement) - # of trees
☐Add Code Button (Show Library Code and Run it)
☐Support Running Code
☐Number of dots should match the number of rows

TODO 8/24
Add duplication
Random forest: take sqrt()
Bagging: Use all features/columns
Use Library to automate the trees
    Add 400/500 trees
    Show best tree + total (bagging) accuracy
    User doesn't create tree
    Explain the majority voting
    Show code
    Add graph to show (# trees vs Accuracy/Time)
User coding
    -Add info button/hover box for the lines of code (explanation)
    -Show Solution button (Practice)
    -Show solution in example/improvement

TODO 8/31
✓Think about how to add audio files / replace textboxes
Add python export
Coding Page
    ✓Add Tutorial to Coding Page to explain the steps of the code
    ✓Change examples files or add their own file (File Explorer)
    ✓Show actual source code in .txt file
        -How to download python
        -How to download libraries
More Info
    ✓Open external pdfs with system viewer
    ✓Open GenerateDecisionTree.pdf, ImproveDecisionTree.pdf
    Add short description of each file by the button
KNN Sections
    Introduction
    Example
        ✓change K (like cross validation)
        ✓Show dots not table
        ✓work with only two features/dimension (add more later)
    Coding
    More Info
KNN Other
    Generate / Find Data (Tshirt size ~ person height and weight)

TODO 9/7
✓Add comments to the code files
✓Change "Select File" to "Select Data File" in Coding Page
Reminder:
    Visualize Tree
    ✓Audio
KNN Example
    ✓Pick simpler data (2 classes and less data)
    ✓The classes should be better distributed and separated easily
    ✓Look at the "KNN Decision.docx" data sets (insert and reduce)
    ✓Add Legend
    ✓Let user pick k

    Functionality:
        ✓Add White Dot/Square - show how it changes with different k values
        Let user manually pick class for white dot

TODO 9/16
-Coding Page: Only allow the block to be placed if it is in the correct position
-User should have choice between text or sound. If click sound button -> hide text and play sound instead.
-Group the menu into categories
    -Classical ML Models: KNN, Linear Regression, Logistic Regression
    -Modern ML Models: Decision Tree, SVM

-Linear Regression
    Introduction
    Example
        Let user pick the slope and intercept
        Show the error
        The user will try to minimize the error
    Extensions
        Polynomial Regression (ax^n + bx^n-1 ... + k) (Quadratic - think about n-power)
        Let user draw the line
        Show the error
        The user will try to minimize the error
    Coding
        Pick exponent and show accuracy
    Further Reading

    Look at data sets in LR Design.docx

    Let Dr.Zhu know if we will have a meeting on Sunday

TODO 9/27
    Linear Regression
        Example:
            Implement Linear Regression algorithm and animate the model training
            Compare user's solution with computer's solutions
        Add multiple variables (show it as 2d graphs)
        Show polynomial equation for the coding page
        
    KNN
        Fix KNN Example 
        Change selection points from gray to the class color
        Remove duplicate rows in iris.csv (change in code, instead of data file)

TODO 10/4
    Linear Regression
        Implement Linear Regression Algorithm (also keep sklearn.LinearRegression - use together)
            Compare user line with computer's line
            Animate training of the model in Example
            Add show me ML results
        Further Reading
            CrossValidation.pdf
            LinearRegression.pdf
            ExtensionLinearRegression.pdf (coming soon)

        STILL NEED TO DO:
            Add R or R^2 value for regression
            Look at examples on different ways to visualize (multiple features)
            Extension
                Select different features with Lasso or SelectKBest (best feature subset selection) algorithm (use library) - try to implement
                Add second example file to Example that is quadratic
    
    Writing or Video (Extra) - if have time
        Libraries used
        IDE to run python
        How to install libraries

TODO 10/11
    Show ML Results error as well as user error
    Remove hardcoded width/height from __init__ in Linear class
    Organize examples into folders
    Add more linear regression examples from LR Design.docx (5-6) including quad equation:
        https://lionbridge.ai/datasets/10-open-datasets-for-linear-regression/
        https://www.smackslide.com/slide/linear-regression-example-data-y4bosz


    Start logistics regression

TODO 10/18
    Linear Example
        Experiment with changing eq/error color/position (grouping by eq or algorithm)
        Extension: power 2, lasso, subset selection
        Add compModel to power k
    
    Logisitic Example:
        Add Coding (Should have 3 data files)
    
    Coding page
        Fix font / change appearance of "Select Data File" Menu
        
    Modern Model
        add Neural Networks(handwriting detection)
    
    If time permits
        New page: data cleaning
        New page: introduction to machine learning (supervised/unsupervised learning)
        New page: Introduction to python / library

TODO 10/25
    Think about normalization
    Linear
        Lasso Regression with all features vs Simple Linear Regression with all features
        Let user pick x column - pick multiple features - show error (Coding page)

    
    Logistics
        Fix the Y column - outcome column
        Let user pick x column - pick multiple features - show error (Coding page)
        Clean Data - replace null values with average of same class
        Add new examples - depression (yes or no)
        
TODO 11/8
    Comparsion models -- see email below -- remove decision tree
    run 50 times - take avg / std / range
    Don't create table - graph the avg and range together - include std somehow?

    How to create correlated variables
        x1 ~ N(0,1)
        x2 ~ N(0,1)
        x3 = p*x1 + sqrt(1-p^2)*x2
        y1 = u1 + o*x1
        y2 = u2 + o*x2

        p - correlation coefficient (0.5)
        u - mean
        0 - standard deviation

    Email
    I would love to compare different classification algorithms at different scenarios. We only consider two classes.
    The possible algorithms are 1) KNN with k = 1; 2) KNN with no limitation of K (algorithm can determine K); 3) logistic algorithm; //4) not decision tree - maybe SVM
    We consider the following scenarios

    Scenario 1. There are 20 training cases in each of two classes. The cases within each class are uncorrelated random normal variables with a different mean in each class
    Scenario 2. There are two features. The data for each feature is t-distribution.  Each class has 50 training cases.
    Scenario 3. The data are generated from a normal distribution, with a correlation of 0.5 between the features in the first class and correlation of -0.5 between the features in the second class.

    You may random generate more data and run the cross validation to see the results from different classification algorithms.

    Linear Regression
        Data - add outlier data - add some pts in the top right corner (shifting / rotating ML Results)

TODO 11/16
    Continue working model comparsion (from 11/8)
    Start working on SVM

TODO 11/22
    Logisitic Comparsions - need to be comparable to KNN (x,y) -> label (z coordinate)
        Change Logisitic to Classifier
    Find Data where each model outperforms the others

    SVM
        1 Hard line and 2 soft line
        Improvement: Kernel functions
        Multiple classes - iris (start with 2)
    
    Start Neural Networks
        Visualize training (show forward and back propagation)

TODO 11/29
    Add library for SVM - one possibility: libSVM
        www.csie.ntu.edu.tw/~cjlin/libsvm
    Create document how and which library to use

TODO 12/13
    Website
        Add about page - how to use and what this is about
        Allow download of python application
            In Future: Same as python application
        Think about using username and password
        Add a way to download the further resources
            Remove the links from the application
        Hosting??
    
    Second Priority: SVM, ANN
        Add acknowledgements to data sources.
        Models should reflect the book (chapters)
            KNN(2), Linear(3), Logistic(4), DT(5),SVM(6), ANN(7)

    Last Priority: Writing (May go in book or website)
        Write demo so Zhu and decide where it goes.
        Maybe have an entire lab section to book
        
        1) Introduce libraries we will be using in Python (pandas, numpy - not ML Libraries)
        2) How to install libraries
        3) Specific code translations from book's math
    
    General Project
        Move this document to trello
        https://trello.com/b/h3OzsIph/teachapp-python

TODO 1/3
    1. Talk to HPU about website credentials
    2. Upload code to website
    3. Upload the 4 models
    4. Find Candidates

TODO 1/10
    1. Add kernels: polynomial, RBF(exponential)
    2. Test adding new data
    3. ANN

TODO 1/17
    -Acc/Eq/Error text - make font larger
    -Make dots larger (Linear, DT, KNN)
    -Change dot color (Linear - red line/dots)
    -Add increment function (increase k)
    -Coding Linear - pick n
    -show training error (Linear)
    -rotate y axis text
    -add cross validation to coding (KNN - best K)
    -Compare with ML Library to test results (especially logistic, maybe SVM - don't worry about the animation)
    -DT: room view label color needs to match table color
    -DT Bagging:  reset graphics when saving tree
    -Check Linear/Quadratic fitting algorithm (Error - Comp Page)
    -fix axis name ("petalLength" -> "Petal Length")
    -Logistic - add ML visualization/comparsion
    -add more data, clean up data (KNN - only has 1)
    -Hide Comparsion model for now

    -hide accuracy before training (SVM)

TODO 2/14
    - Coding page: allow user to select svm kernel
    - SVM: choose a complicated data set (Maybe medical data)
    - SVM: Add legend

    - Research: Mathlib - repeat their work by running their code.
    - We can try their work with k-mean + svm

TODO 3/10
    - Research
        - Need to obtain their data (email them)
        - Repeat their work with polynomial / rbf kernel
        - Use modified SVM
    - TeachApp
        - Add explanation that 'Unfit SVM' is not linearly separable and requires a different kernel. 

TODO 3/17
    -TeachApp
        - Focus Entirely on ANN simulations
    - Research Game Plan
        - Wait for the data and modify SVM (Based on 3/10 Meeting)
        - Contact Fujitsu for their data and modify SVM
        - Use OPNet/algorithm to generate traffic pattern and learn Network Activity to improve resource allocation

TODO 3/24
    Continue ANN
    Some datasets:
        https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v2
        https://knowledgedefinednetworking.org/
        Topics
            Understanding the Network Modeling of Computer Networks:
            Unveiling the potential of GNN for network modeling and optimization in SDN

TODO 3/31
    - Add MNIST data set to train ANN (Example)
    - Next one: user can create their data set.
    - Look at Dataset (from 3/24) and discuss on Sunday (4/4) - 10am.
        - Find out: Features? Model? Missing data? Figures? Submit paper?
        - Try to replicate work?
        
"""
