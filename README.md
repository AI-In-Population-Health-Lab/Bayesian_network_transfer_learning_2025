# BNTL_update_version
# 1. Introduction 

The goal of this project is to increase the re-use of computable biomedical knowledge in probabilistic formalisms.   
In the field of transfer learning, which is a sub-area of machine learning, these differences between settings are referred to as heterogeneity. Unlike traditional machine learning, transfer learning explicitly distinguishes between a source setting, from which we develop a model M that we would like to re-use, and a target setting, for which we use or adapt M, often because the target setting has insufficient data to develop a model de novo. Researchers distinguish transfer learning algorithms by what kinds of heterogeneous transfer learning scenarios (or simply heterogeneous scenarios) they can handle.  

This project created the Bayesian Network Transfer Learning (BN-TL) algorithm to re-use of source model, such as  influenza,  learned from electronic medical record (EMR) data to predict the target data set. (for more info see: Dr. Ye's proposal approach & her paper *Transfer Learning For Bayesian Case Detection Systems*)



# 2. Quick Start  

*If you have not install JAVA JDK yet, please google it and install first*

## 2.1 requirement 
those two packages are the core of this project. 
- [Weka package](https://waikato.github.io/weka-wiki/)  It provides data format structure, the Bayes Network Structure, classifier, evaluation methods.  
- [Jsmile package](https://support.bayesfusion.com/docs/Wrappers/)It is encoding the Bayes Network for depth of probability calculation. 


## 2.2 Set up 
Please check the video in tutorials9 folders. 

###  Instructions 
1.  Download the zip file from the [github](https://github.com/AI-In-Population-Health-Lab/BNTL_multiSource.git) / or `git clone https://github.com/AI-In-Population-Health-Lab/BNTL_multiSource.git
2.  Intellj IDEA--> open project--> find the directory for the unzipped file
3.  Set up JDK as video shows( Java 1.8, 11, or 18 is good)
4.  Go to `runnerBNTL`  class to run--> Edit configurations--> Modify options-->add VM options (under Java tab)
5.  add `-Djava.library.path=your lib path` --> apply-->Ok

## 2.3 Jar quick start  
- Install JAVA [JDK 18](https://www.oracle.com/java/technologies/javase/jdk18-archive-downloads.html) (you may use other version as well)
```
java version "18.0.2.1" 2022-08-18

Java(TM) SE Runtime Environment (build 18.0.2.1+1-1)

Java HotSpot(TM) 64-Bit Server VM (build 18.0.2.1+1-1, mixed mode, sharing)
```


- Smile package for Java [Link](https://download.bayesfusion.com/files.html?category=Academia)
```
jsmile-academic-2.0.10
```

### 2.3.1 How to Run   

### 1. Download zip file 
### 2. use Intellj IDEA open the project 
### 3. Go to src-> test_start class
### 4. Edit Configruation-->Modify Options-->VM options-->`-Djava.library.path=`

---

### 1. download the BNTL.jar package 
### 2. Open the terminal/ cmd windows 

```
java -jar BNTL_MultiSource.jar
```

After typing this, it will provide this content
```
=====================================BNTL Project===========================
This project created the Bayesian Network Transfer Learning (BN-TL) algorithm to re-use of source model, 
such as influenza, learned from electronic medical record (EMR) data to predict the target data set. 
(for more info see: Dr. Ye's proposal approach & her paper Transfer Learning For Bayesian Case Detection Systems)
============================================================================


-----------------------------------Instruction------------------------------
Please check the input parameters: [file_path] [configuration_file] [on/off] [unadjust] [ratio] [on/off]

[Parameter 0: smile_path] [Parameter 1: file_path]
[Parameter 2: configuration_file]
[Parameter 3: unadjust weight] [Parameter 4: ratio weight]
[Parameter 5: KL weight method: on/off]

For instance: 
java -Djava.library.path=[smile_path] -jar xxxx.jar file_path config_file off unadjust raito off

```

### 3. Prepare input folder 

#### 3.1 File type 
- data must be in .arff format. 
- model must be in .bif, .xml, .xdsl format.

#### 3.2 structure of input folder 
	folder_name
		utility
			your configurationFile.txt
		sourceData
			sourceData.arff
		sourceModel
			source_learn_model.bif
		targetData
			target_test_data.arff
			target_train_data.arff
		targetModel
		result
			intermediate folder-->to store result. 
		log


#### 3.3 parameters for configurationFile.txt 

```

-----Source Part-----
sourceDataLoc=
sourceModelLoc=
sourceDataName=source_data.arff
sourceLearnedModelName= 1.bif,2.bif,...
sourceDataSize=1661
-----Target Part-----
targetNodeName=diagnosis
targetTrainDataSize=21
targetTestDataName=test_data.arff
targetTestDataSize=344
targetModelLoc=path--->targetModel/
targetDataName=train_data.arff
targetDataLoc=path-->targetData/
-------feature-------
modelLearningApproach=NB
featureSelectionApproach=IG001

-------Path------
temporaryFileName=temp.xml
experimentFileLoc= path to folder
utilityLoc=path-->/utility/
resultProbName=prob.csv
resultAUCName=auc.csv
logName=configuration_file.txt
resultLoc=path--->result/
logLoc=path-->log/
hashCodeName=hashCode.xml


------------KL----------(if you want to perform KL section)

trueKL=1.8224292702088067  

trueBFTable=hypoxemia,1.0E-9;reported_fever,1.0E-9;influenza_lab_positive,1.0E-9;age_group,1.0E-9;diagnosis,1.0E-9;unspecified_cough,3.2417270528642096E-6;nasal_swab_order,1.0E-9;   

KL_targetData_learnedSourceModel=1.2568181979662876  

nodeKLTable_targetData_learnedSourceModel=diagnosis,0.0045124560875614075;unspecified_cough,0.09813077790883455;reported_fever,0.4261776773290663;age_group,0.01770522273709965;hypoxemia,0.47951291320078326;nasal_swab_order,0.114442540587127;  

KL_targetData_trueSourceModel=1.660745666413443  

nodeKLTable_targetData_trueSourceModel=diagnosis,0.03595232239858527;age_group,0.0281376732895748;nasal_swab_order,0.05359926700974576;unspecified_cough,0.019446531674810627;reported_fever,0.4892853435516517;influenza_lab_positive,0.09896658196551227;hypoxemia,0.8059805217479895;  


```

- performance baseline, you need to have true model for both source and target. 

- run unadjust & ratio performance, you need to have Source Part, Target Part, Feature Part, and Path part. 

- Run KL options, you need to have KL Part that includes in configuration file. 

- For multi-source processing, `sourceLearnedModelName= 1.bif,2.bif,...`  add `, model.bif`. 

### 4. Run at terminal/ cmd 

since we use Jsmile package, we need to manually at library to our local machine. 

After you download the Jsmile-2.0.10, extract it and put it into folder (*windows/ mac have different file*)

#### 4.1 Run

```
java -Djava.library.path=[jsmile path] -jar BNTL.jar [parameter 0] [parameter 1] [parameter 2] [parameter 3] [parameter 4] [parameter 5]
```



### 5. Result 
result will be stored in `result/intermediate` folder and target model will be in `targetModel`.



### 6. Limitation 

- The max parent <= 6 
- batch processing
- data format in .arff 





## 2.3 Outline of the Project 

### Overview 

#### 1. transform learning performance  
Five categories with different `transferLearningWeight`  
source_true_model  &  Target Training Data  
1. KL_targetData_trueSourceModel  
2. nodeKLTable_targetData_trueSourceModel
3. unadjust
4. ratio
5. trueBayesFactor

#### 2. Evaluation 


### Folders 
There are 4 folders 
1. `000influenza_ac_topazac_slc_topazslc` 
		This folder contains source data, source model, target data, target model to run experiments. 

2. `lib` 
		This folder contains packages, which have been used for this project. 
		The most important is weka and Jsmile (libjsmile.jnilib)

3. `out` 
		This is class output folder.  For intellj IDEA, it is called out.  For eclipse, it is called bin. 

4. `src` 
		The most import folder because all java code are in this folder.  

		1. edu.pitt --> contains Jsmile probability 
		2. Utility--> BNTL  the core code for running the experiment 
		3. weka --> revise version weka package to avoid some exceptions, comparing the original weka package. 
		4. `runnerBNTL`  a class to run `BNTL` method. 



# 3. Methods  

## 3.1 `runnerBNTL` main method  

For this project, the section need to pay an attention is `runOneConfig(String configLoc, String fileName, int folderNum)` as start place. 

```Java
public static void runOneConfig(String configLoc, String fileName, int folderNum) throws Exception{
    // ......

    // start here  Based Line performance 
    printBaseLinePerformance();

    // important section 
    getModels(configFileName);

    //......

}


// the most important part for runnerBNTL section. 

// running experienment method to get BNTL methods. 
public static void getModels(String configFileName) throws Exception{

    //......

    // weka arff load data into instances. 

    // running the experiement 
    // one experienment 
//----------------------------------------------------------------------
//******** The most important part in runnerBNTL*************************
// all transfer learning algorithms is in this new BNTL class.

// pay attention to transferLearningWeight
// since 2nd, 3rd,4th,5th's experiences, the only difference is transferLearningWeight.  Ohter parameters are keep same. 
    bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable));

//----------------------------------------------------------------------

    // build classifier
    bn.buildClassifier(instances);
    // evaluation
    printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);

    // second one
    
    // it keeps same pattern so just focus on first one. 

    // third one

    // fourth one

}

```

## 3.2 `BNTL` class  
It is under Utility package.  

### 3.2.1 Constructor  
Read all configuration from files, and get source_true_model.


Those parameters are global variables. It passes from `runnerBNTL` & `bn.setSearchAlgorithm(new BNTL())`.  The only difference is `String adjustMethod`  
1. KL_targetData_trueSourceModel  
2. nodeKLTable_targetData_trueSourceModel
3. unadjust
4. ratio
5. trueBayesFactor



``` Java
public BNTL(String configFile, String scoreMethod, String adjustMethod,String startNetwork,  double trueKL1, HashMap<String,Double> trueBFWeightTable1,double avgKL1, HashMap<String,Double> avgBFWeightTable1){

 }

```


### 3.2.2 Core Section  

```Java

// bayesNet is targetData initialized 
public void buildStructure(BayesNet bayesNet, Instances targetTrainingData){

    // There are 3 parts for this method 
    // Part A. clean source model 

    // Part B. grow from source_true model and Searching models

    // Part C. Evaluation 

}

```

#### Part-A Clean Source Model  

1. remove unobserved node  

`cleanSourceBN(source_BayesNet, targetTrainingData)`  

2. Inject target features into sourceModel  

`injectTargetSpecificFeature(source_clean_BayesNet, targetTrainingData)`  


#### Part-B Grow and Search  

1. initialization  

Jsmile Bayes Network--> convert weka Bayes Network to Jsmile Bayes Network.  
`etwork net = new Network()` & `net.readFile()`  

initial variables  
`readCases()`  ---> read arff data into multi-dimension array. 

`initVariables()` --->  all variables.  nodeInfo/ FileCaseRecord.  


2. Scoring  

- Score  

Score structure:  
`scoreNode()` --> `deriveNodeProbs_Prior_Data()`-->`getPriorProb()`  


`scoreNode()`  main method to calculate each node's score.  

**The Most Important**  

`deriveNodeProbs(casei, node, a, b)`  


`deriveNodeProbs_Prior_Data(casei, node,tempAdjustPriorSampleSize)`  

In those methods are to get `Nijk` formula from Yeye's paper. 


`getPriorProb` --> use Jsmile Net to calculate probabilities.   
`netWork.probabilities()` --> under the `edu.pitt.isp.sverchkov.smile` package.


- CPT  

`getCPTCombineModelData`


3. Evaluation



# 4. Resources   











