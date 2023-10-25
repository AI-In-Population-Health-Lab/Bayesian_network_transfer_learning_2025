
import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.List;

// weka package
import edu.pitt.isp.sverchkov.smile.SMILEBayesNet;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.bayes.net.EditableBayesNet;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

// import Utillity BNTL
import Utility.BNTL;
import Utility.Utility;

import edu.pitt.dbmi.dataset.ConvertBIFtoGENIE;
import weka.gui.graphvisualizer.GraphVisualizer;

import javax.swing.*;



// smile package
import smile.Network;

// date

public class runnerBNTL {
//====Constant ================
    static String configFileName;

    static String sourceModelLoc,sourceLearnedModelName,sourceLearnedCleanedModelName, sourceTrueModelName,
            sourceTrueCleanedModelName, sourceCleanInjectModelXML,
            sourceSimulateDataLoc,sourceSimulateDataName,sourceDataLoc,sourceDataName,sourceDataSize,
            targetModelLoc,targetTrueModelName,targetLearnedModelName, targetDataLoc,targetDataName,targetNodeName,
            targetTestDataName,resultLoc,resultProbName,resultAUCName,resultCalibrationName,
            utilityLoc,temporaryFileName,hashCodeName,logLoc,logName, targetClassName;

    static HashMap<String,Double> learnedSource_nodeKLTable;
    static HashMap<String,Double> trueSource_nodeKLTable;
    static String featureSelectionApproach, modelLearningApproach;

    static HashMap<String,Double> trueBFWeightTable;
    static double trueKL;
    static double KL_targetData_learnedSourceModel, KL_targetData_trueSourceModel;
//===================================================================================

    static String targetFinalModelName;


    static String transferLearningApproach, transferLearningWeight;

    static double avgKL;
    static HashMap<String,Double> avgBFWeightTable;

//============================================
    static PrintWriter printoutAUC;


// multiple version
    static ArrayList<String> sourceClearnInjectModelXDSL;


//------------------------------ Main Function ---------------------------------------//
    /*
     */
    public static void main(String[] args) throws Exception {
        // TODO Auto-generated method stub

//        new smile.License("SMILE LICENSE d918bddd 871cb16c 3d01a731 " +
//                        "THIS IS AN ACADEMIC LICENSE AND CAN BE USED " +
//                        "SOLELY FOR ACADEMIC RESEARCH AND TEACHING, " +
//                        "AS DEFINED IN THE BAYESFUSION ACADEMIC " +
//                        "SOFTWARE LICENSING AGREEMENT. " +
//                        "Serial #: 75ivcaomfx4xfthy077ica6bx " +
//                        "Issued for: Ye Ye (yeyewy@gmail.com) " +
//                        "Academic institution: University of Pittsburgh " +
//                        "Valid until: 2023-09-29 " +
//                        "Issued by BayesFusion activation server",
//                new byte[] {103,99,55,-38,-42,24,-11,-31,17,-23,-92,-121,-45,-108,-70,90,
//                        -123,-7,91,-2,-11,-51,-108,-44,-35,33,93,-11,-53,-72,31,-15,
//                        23,43,-31,-65,81,101,47,-123,95,-40,-4,-101,90,98,-101,-11,
//                        -27,49,-121,-38,-92,-97,73,39,-70,-40,95,-21,13,-111,-83,-98}
//        );

        new smile.License(
                "SMILE LICENSE 27423093 79d0cdab 41c62a7f " +
                        "THIS IS AN ACADEMIC LICENSE AND CAN BE USED " +
                        "SOLELY FOR ACADEMIC RESEARCH AND TEACHING, " +
                        "AS DEFINED IN THE BAYESFUSION ACADEMIC " +
                        "SOFTWARE LICENSING AGREEMENT. " +
                        "Serial #: bz8ak08a5jf2cmdzl2lsvk77y " +
                        "Issued for: Cat (johncxsong@gmail.com) " +
                        "Academic institution: University of Pittsburgh " +
                        "Valid until: 2024-04-27 " +
                        "Issued by BayesFusion activation server",
                new byte[] {
                        -99,103,-1,87,104,78,-4,-78,47,-101,-95,-57,-27,-47,-11,21,
                        -27,0,-6,67,-98,79,-8,-23,-105,-123,34,83,-127,-47,73,11,
                        -117,39,-13,-90,-46,76,-115,-124,-114,-39,-108,23,-57,67,87,54,
                        -109,-25,114,61,45,113,-36,-21,25,-61,56,-80,-126,-123,58,40
                }
        );


        // [filepath, configurationFile, on/off, unadjust,ratio,on/off ]
        System.out.println("=====================================BNTL Project=========================================================");
        System.out.println("This project created the Bayesian Network Transfer Learning (BN-TL) algorithm to re-use of source model, " +
                "\nsuch as influenza, learned from electronic medical record (EMR) data to predict the target data set. " +
                "\n(for more info see: Dr. Ye's proposal approach & her paper Transfer Learning For Bayesian Case Detection Systems)");
        System.out.println("==========================================================================================================");
        System.out.println();
        System.out.println();


        if(args.length!= 5){
            System.out.println("-----------------------------------Instruction-----------------------------------------------------------");
            System.out.println("Please check the input parameters: [file_path] [configuration_file] [unadjust] [ratio] [on/off]");
            System.out.println();
            System.out.println("[Parameter 1: file_path] [Parameter 2: configuration_file]\n" +
                    "[Parameter 3: unadjust weight] [Parameter 4: ratio weight]\n" +
                    "[Parameter 5: KL weight method: on/off]---> to perform this. You must have KL number in your configuration file");
            System.out.println();
            System.out.println("For instance: ");
            System.out.println("java -Djava.library.path=[smile_path] -jar xxxx.jar file_path config_file unadjust raito off");
            System.out.println("--------------------------------------------------------------------------------------------------------");

            return;
        }


        String filePath= args[0];
        String configurationFile = args[1];

        // transformed learning package
        String unadjust = args[2];
        String ratio = args[3];
        String KL = args[4];

        // Experiment folder
//=======================================================================================================================
        // relative
        //String folderExperimentLoc = "./000influenza_ac_topazac_slc_topazslc/";

        //path for jar
        String folderExperimentLoc = filePath;
        String fileName = configurationFile;
        // /Users/johnsong/documents/ye_lab/bayesnet_tranforming/000influenza_ac_topazac_slc_topazslc


       //String fileName = "folder10_Influenza_CFS_acmodel_0810_topazac_size50_slcmodel_0810_topazslc.bif_targetDataSize_1000-IG";
//===================================================================================================================================

        // call configuration and run experiment from this function
        // folderNum is redundant.
        String configLoc = new String(folderExperimentLoc+"/utility/");
        runOneConfig(configLoc,fileName,unadjust,ratio,KL);

    }



//-------------------------------Run Configuration and Experiment----------------------------------------------------------------------------//
    public static void runOneConfig(String configLoc, String fileName,String unAdjust,String ratio,String KL) throws Exception{

        // This section is log section
        // 这里就是写了一个文件，记录一下。因为对实验结果不影响，所以放到了一个function 里面。
        // this function assign global variable.

        nothingFunction(configLoc,fileName, KL);
       //------------ ----------------------------

//========================Based Line performance===================//
    /*     1. Based Line performance:
          Four categories:
            1. target_True_model &  target test DataSet
            2. target_Learn_model & target test DataSet

            3. source_clean_true model[delete nodes that do not appear in target DataSet] & target test DataSet
            4. source_clean_Learn model & target test DataSet
    */



//==========================Start to Run model==========================================//
     /* 2. transform learning performance:
     Training section:
        start  source_true_model  &  Target Training Data  IN didfferent [transferLearningWeight]

        1. KL_targetData_trueSourceModel
        2. nodeKLTable_targetData_trueSourceModel
        3. unadjust
        4. ratio
        5. trueBayesFactor
      */
        //configFileName = configLoc + fileName + ".txt";
        getModels(unAdjust,ratio,KL);

        //Evaluation
//========================================================================//

        printoutAUC.close();
    }

//-----------------------------------------------------------------------------------------------------------//



//----------transform learning performance Running Part-----------------------------------------------------//

    public static void getModels(String unAdjust, String ratio, String KL) throws Exception{

// feature selection  IG==================================================
        try {
            ArffLoader arff = new ArffLoader();
            arff.setFile(new File(targetDataLoc+targetDataName));

            Instances instances = arff.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);

            Instances filtedData = instances;
            if (featureSelectionApproach.equals("CFS")){
                filtedData  = FeatureSelection.useFilterCfsSubsetEval(instances);
            }
            else if (featureSelectionApproach.equals("IG")){
                System.out.println("feature selected target training data:");
                filtedData  = FeatureSelection.useFilterInfoGain(instances, new Double("0.0001"));
            }

 //=============================================

// source_Learn data & target training data==================================
            //get target trainng data
            //slcmodel_0810_topazslc_seed1191167696_size1000.arff
            instances = filtedData;

            //acmodel_0810_topazac_seed-665150065_size50-IG-K2.bif
            System.out.println("Source model is: " + sourceLearnedModelName);

            System.out.println("Target training data is: " + targetDataName);

            System.out.println("===================Transformation ==========================");

            transferLearningApproach = new String("priorModelApproach");

// run source_clean_model experiment
            //===== start  source_learned_model  &  Target Training Data=============================================================================//

            // start search from learned source model
            String startNetworkName = sourceLearnedModelName;
            printoutAUC.println("startNetworkName:" + startNetworkName);

            // multiple-version
            // if netWorkList = 1. it is single source model
            // if networkList >1, it is multi-source model
            ArrayList<SMILEBayesNet>  netWorkList = readNetwork();



// 3. unadjuested

            if(unAdjust.equals("unadjust")){
                unadjust(instances,startNetworkName,netWorkList);
            }

            if(ratio.equals("ratio")){
                ratio(instances,startNetworkName,netWorkList);
            }

            if(KL.equals("on")){
                // 5. trueBayesFactor
                trueBayesFactor(instances,startNetworkName,netWorkList);
                // 1. KL_targetData_learnedSourceModel
                KL_targetData_learnedSourceModel(instances,startNetworkName,netWorkList);
// 2. nodeKLTable_targetData_learnedSourceModel
                nodeKLTable_targetData_learnedSourceModel(instances,startNetworkName,netWorkList);

            }



// run source_true_model experiment
           // runnerBNTL.sourceTrue(instances);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 1. KL_targetData_learnedSourceModel
    public static void KL_targetData_learnedSourceModel(Instances instances,String startNetworkName,ArrayList<SMILEBayesNet>  netWorkList) throws Exception {
        // 1. KL_targetData_learnedSourceModel
        BayesNet bn = new BayesNet();
        transferLearningWeight = new String("KL_targetData_learnedSourceModel");
        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable,netWorkList));
        bn.buildClassifier(instances);
        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);
    }

    public static void nodeKLTable_targetData_learnedSourceModel(Instances instances,String startNetworkName,ArrayList<SMILEBayesNet>  netWorkList) throws Exception{
        // 2. nodeKLTable_targetData_learnedSourceModel
        BayesNet bn = new BayesNet();
        transferLearningWeight = new String("nodeKLTable_targetData_learnedSourceModel");
        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable,netWorkList));
        bn.buildClassifier(instances);
        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);
    }

    /**
     * intances--input data, sourcelearnedModel as start, netWorkList is the multi-source model.[1.xdsl,2.xdsl,..] the first is the main source model.
     * @param instances
     * @param startNetworkName
     * @param netWorkList
     **/

    public static void unadjust(Instances instances,String startNetworkName,ArrayList<SMILEBayesNet>  netWorkList) throws Exception{
        BayesNet bn = new BayesNet();
        transferLearningWeight = new String("unadjust");
        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                trueKL,trueBFWeightTable, avgKL,avgBFWeightTable,netWorkList));
        bn.buildClassifier(instances);
        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);

    }

    public static void ratio(Instances instances,String startNetworkName,ArrayList<SMILEBayesNet>  netWorkList) throws Exception{
        BayesNet bn = new BayesNet();

        transferLearningWeight = new String("ratio");
        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                trueKL,trueBFWeightTable, avgKL,avgBFWeightTable,netWorkList));
        bn.buildClassifier(instances);
        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);
    }

    public static void trueBayesFactor(Instances instances,String startNetworkName,ArrayList<SMILEBayesNet>  netWorkList) throws Exception{
        BayesNet bn = new BayesNet();
        transferLearningWeight = new String("trueBayesFactor");
        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                trueKL,trueBFWeightTable, avgKL,avgBFWeightTable,netWorkList));
        bn.buildClassifier(instances);
        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);

    }




    // Source_learned model


    // load multi-source models.
    // this method only contain two network.
    // will modify later for multi-source in one folder. 
    public static ArrayList<SMILEBayesNet> readNetwork(){
        // add network list,
        // since it is smile network,  create an array list
    // Smile Network
        ArrayList<SMILEBayesNet>  netWorkList = new ArrayList<>();

        for(String temp: sourceClearnInjectModelXDSL){

            Network net = new Network();
            net.readFile(sourceModelLoc+temp);

            // Smile Network
            boolean convertIDs = false;
            SMILEBayesNet network = new SMILEBayesNet(net, convertIDs);
            netWorkList.add(network);

        }

        return netWorkList;
    }




    public static void cleanSourceLearnedModel() throws Exception{
        BayesNet source_BayesNet = new BIFReader();
        try {
            ((BIFReader) source_BayesNet).processFile(sourceModelLoc+sourceLearnedModelName);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        Instances targetData = new Instances(new BufferedReader(new FileReader(targetDataLoc+targetDataName)));
        BayesNet source_learned_clean_BayesNet = cleanSourceBN(source_BayesNet, targetData);
        //	System.out.println("Source_clean:");
        //	System.out.println(source_clean_BayesNet.graph());
        PrintWriter printout = new PrintWriter(new File(sourceModelLoc+sourceLearnedCleanedModelName));
        printout.println(source_learned_clean_BayesNet.graph());
        printout.flush();
        printout.close();
        ConvertBIFtoGENIE conv = new ConvertBIFtoGENIE();
        conv.runner(sourceModelLoc+sourceLearnedCleanedModelName, "", utilityLoc+hashCodeName, "xdsl");
    }

    public static void cleanSourceTrueModel() throws Exception{
        BayesNet source_BayesNet = new BIFReader();
        try {
            ((BIFReader) source_BayesNet).processFile(sourceModelLoc+sourceTrueModelName);


        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        //slcmodel_0810_topazslc_seed1191167696_size1000.arff
        Instances targetData = new Instances(new BufferedReader(new FileReader(targetDataLoc+targetDataName)));

        BayesNet source_true_clean_BayesNet = cleanSourceBN(source_BayesNet, targetData);
        //	System.out.println("Source_clean:");
        //	System.out.println(source_clean_BayesNet.graph());
        PrintWriter printout = new PrintWriter(new File(sourceModelLoc+sourceTrueCleanedModelName));
        printout.println(source_true_clean_BayesNet.graph());
        printout.flush();
        printout.close();
    }

    /**
     * remove a node from source model when the node does not appear in target data
     * @param bayesNet
     * @param targetInstances
     * @return
     * @throws Exception
     */
    public static BayesNet cleanSourceBN(BayesNet bayesNet, Instances targetInstances) throws Exception{

        // save model as temp.xml  and read from it.
        System.out.println("------Cleaning source specific features");
        PrintWriter printout = new PrintWriter(new File(utilityLoc+temporaryFileName));

        printout.println(bayesNet.graph());
        printout.flush(); printout.close();
        BIFReader reader = new BIFReader();
        EditableBayesNet tempNet = new EditableBayesNet(reader.processFile(utilityLoc+temporaryFileName));
        System.out.println(utilityLoc+temporaryFileName);

        //get attribute list of the target data
        ArrayList<String> attributeList = new ArrayList<String>();
        int numAttribute = targetInstances.numAttributes();

        for (int i=0; i<numAttribute; i++){
            String oneNodeName = targetInstances.attribute(i).name();
            attributeList.add(oneNodeName);
        }

        //If a node does not appear in the attribute list of the target data, add it to deleteNodeList
        ArrayList<String> deleteNodeList = new ArrayList<String>();
        int numNodes= tempNet.getNrOfNodes();
        for (int iNode=0; iNode<numNodes; iNode++){
            String nodeName =  tempNet.getNodeName(iNode);
            if (!attributeList.contains(nodeName)){
                deleteNodeList.add(nodeName);
            }
        }
        //delete nodes
        for (int i=0; i<deleteNodeList.size(); i++){
            tempNet.deleteNode(deleteNodeList.get(i));
            System.out.println("delete:" + deleteNodeList.get(i));
        }
        return tempNet;
    }

    public static void printOnePerformance(String modelLocation, String modelName, String testDataset) throws Exception{
        //	String testDataset = new String(targetDataLoc+targetTestDataName);
        String model = new String(modelLocation + modelName);
        BIFReader model_BayesNet = new BIFReader();
        model_BayesNet.processFile(model);
        System.out.println(modelName + ":" + evaluateBN(modelName, model_BayesNet, testDataset));
    }



    public static void printBaseLinePerformance() throws Exception{

            // slcmodel_0810_topazslc_seed266811895_size10000.arff
            String testDataset = new String(targetDataLoc+targetTestDataName);

            //slcmodel_0810_topazslc.bif
            // calculate the target True model with target testing data
            String trueTargetModel = new String(targetModelLoc+targetTrueModelName);
            BIFReader ture_target_BayesNet = new BIFReader();
            ture_target_BayesNet.processFile(trueTargetModel);
            System.out.println("AUC_true_target:" + evaluateBN(targetTrueModelName, ture_target_BayesNet, testDataset));




            // calculate the target Learn model with testing DataSet
            String targetOnlyModel = new String(targetModelLoc+targetLearnedModelName);

            BIFReader targetOnlyModel_BayesNet = new BIFReader();
            targetOnlyModel_BayesNet.processFile(targetOnlyModel);
            System.out.println("AUC_target_only:" + evaluateBN(targetLearnedModelName, targetOnlyModel_BayesNet, testDataset));



            // source_True_Clean model[delete nodes which do not appear in target dataset] with target DataSet
            cleanSourceTrueModel();
            String trueSourceModel = new String(sourceModelLoc+sourceTrueCleanedModelName);
            BIFReader ture_source_BayesNet = new BIFReader();
            ture_source_BayesNet.processFile(trueSourceModel);
            System.out.println("AUC_true_source_clean:" + evaluateBN(sourceTrueCleanedModelName, ture_source_BayesNet, testDataset));


            // source_learn_Clean model[delete nodes which do not appear in target dataset] with target DataSet
            cleanSourceLearnedModel();
            String sourceModel = new String(sourceModelLoc+sourceLearnedCleanedModelName);
            BIFReader sourceModel_BayesNet = new BIFReader();
            sourceModel_BayesNet.processFile(sourceModel);
            System.out.println("AUC_source_clean:" + evaluateBN(sourceLearnedCleanedModelName, sourceModel_BayesNet, testDataset));


    }
    /***
     * Note. I edited the SimpleEstimator distributionForInstance(BayesNet bayesNet, Instance instance)
     * Because Bayesian network and instance may not have the same order of attributes.
     * under C:\Users\yey5\Dropbox\YeEclipseThesis\weka-3.7.9-YeExperiment\src\main\java\weka\classifiers\bayes\net\estimate
     * @param modelName
     * @param BN
     * @param testDataset
     * @return
     * @throws Exception
     */

    public static String evaluateBN(String modelName, BayesNet BN, String testDataset) throws Exception{
        Instances testData = new Instances(new BufferedReader(new FileReader(testDataset)));
        testData.setClassIndex(testData.numAttributes()-1);
        //BN.m_Instances.setClassIndex(classIndex);
        int numNodes= BN.getNrOfNodes();
        ArrayList<String> nodeListInBN = new ArrayList<String>();
        for (int i=0; i<numNodes; i++){
            nodeListInBN.add(BN.getNodeName(i));
            //	System.out.println(BN.getNodeName(i));
        }
//		for (int i=testData.numAttributes()-2; i>=0; i--){
//			String oneFeatureName = testData.attribute(i).name();
//			if (!nodeListInBN.contains(oneFeatureName)){
//				testData.deleteAttributeAt(i);
//				System.out.println("test data delete:" + oneFeatureName);
//			}
//		}

        for (int i=0; i<testData.numAttributes(); i++){
            //	System.out.println("test data remaining:" + testData.attribute(i).name());
        }

        Attribute classAttribute = testData.attribute(testData.classIndex());

        Evaluation eval = new Evaluation(testData);
        eval.evaluateModel(BN, testData);
        PrintWriter bnProbResult = new PrintWriter(new File(resultLoc+"intermediate/"+resultProbName.replace(".csv", "")+"-"
                +modelName.replace(".bif", "-targetFilter.csv")));

        bnProbResult.print(classAttribute.name()+",Prob_");


        int nClassAttribute = classAttribute.numValues();

        for (int i=0; i<nClassAttribute-1; i++){
            String classValue = classAttribute.value(i);
            bnProbResult.print(classValue+",Prob_");
        }

        String classValue = classAttribute.value(nClassAttribute-1);
        bnProbResult.println(classValue);

        for (int i=0; i<=testData.numInstances()-1; i++){
            double[] predications = BN.distributionForInstance(testData.instance(i));
            bnProbResult.print(testData.instance(i).stringValue(testData.instance(i).classAttribute()) + "," );
            for (int j=0; j<=nClassAttribute-2; j++){
                bnProbResult.print(predications[j] + ",");
            }
            bnProbResult.print(predications[nClassAttribute-1] + "\n");
        }
        bnProbResult.flush(); bnProbResult.close();

        PrintWriter bnAUCResult = new PrintWriter(new File(resultLoc+"intermediate/"+resultAUCName.replace(".csv", "")+"-"
                +modelName.replace(".bif", "-targetFilter.csv")));
        String AUC= "weightedAUC:" + eval.weightedAreaUnderROC() + "\n weightedPRC:" + eval.weightedAreaUnderPRC() + "\n  AUC:"  ;
        for (int i=0; i<nClassAttribute; i++){
            AUC = AUC + eval.areaUnderROC(i) + ",";
        }
        AUC = AUC + "\n  PRC:";
        for (int i=0; i<nClassAttribute; i++){
            AUC = AUC + eval.areaUnderPRC(i) + ",";
        }
        bnAUCResult.println(AUC);
        bnAUCResult.flush(); bnAUCResult.close();
        printoutAUC.println(modelName+","+AUC);
        printoutAUC.flush();
        return AUC;
    }


    public static void sourcePart() throws FileNotFoundException {
        sourceModelLoc=Utility.getConfig("sourceModelLoc",configFileName);
        sourceDataLoc=Utility.getConfig("sourceDataLoc",configFileName);
        sourceDataName=Utility.getConfig("sourceDataName",configFileName);
        sourceDataSize=Utility.getConfig("sourceDataSize",configFileName);
        sourceTrueModelName=Utility.getConfig("sourceTrueModelName",configFileName);
        sourceCleanInjectModelXML=Utility.getConfig("sourceCleanInjectModelXML",configFileName);
        sourceLearnedModelName=sourceDataName.replace(".arff", "-" + featureSelectionApproach + "-" + modelLearningApproach + ".bif");
        sourceLearnedCleanedModelName = sourceLearnedModelName.replace(".bif", "")+"_cleaned.bif";
        sourceTrueCleanedModelName = sourceTrueModelName.replace(".bif", "")+"_cleaned.bif";


        // multiple version to get list of network
        // use [1.xdsl, 2.xdsl, 3.xdsl....]
        // add by John Song for multi-version
        String temp =Utility.getConfig("sourceCleanInjectModelXDSL",configFileName);
        String [] temp1 = temp.split(",");
        sourceClearnInjectModelXDSL= new ArrayList<>();
        for(String a: temp1){
            sourceClearnInjectModelXDSL.add(a);
        }

        //sourceSimulateDataLoc=Utility.getConfig("sourceSimulateDataLoc",configFileName);
        //sourceSimulateDataName=Utility.getConfig("sourceSimulateDataName",configFileName);

    }
    public static void targetPart() throws FileNotFoundException{
        targetModelLoc=Utility.getConfig("targetModelLoc",configFileName);
        targetDataLoc=Utility.getConfig("targetDataLoc",configFileName);
        targetDataName=Utility.getConfig("targetDataName",configFileName);
        targetNodeName=Utility.getConfig("targetNodeName",configFileName);
        targetTestDataName=Utility.getConfig("targetTestDataName",configFileName);
        targetTrueModelName=Utility.getConfig("targetTrueModelName",configFileName);
        targetLearnedModelName=targetDataName.replace(".arff", "-" + featureSelectionApproach + "-" + modelLearningApproach + ".bif");
        targetClassName = Utility.getConfig("targetClassName",configFileName);
        // targetClassname = ""
        if (targetClassName==null){
            targetClassName = new String("class");
        }
    }

    public static void logPart() throws FileNotFoundException{
        resultLoc=Utility.getConfig("resultLoc",configFileName);
        resultProbName=Utility.getConfig("resultProbName",configFileName);
        resultAUCName=Utility.getConfig("resultAUCName",configFileName);
        utilityLoc=Utility.getConfig("utilityLoc",configFileName);
        temporaryFileName=Utility.getConfig("temporaryFileName",configFileName);
        hashCodeName=Utility.getConfig("hashCodeName",configFileName);
        logLoc=Utility.getConfig("logLoc",configFileName);
        logName=Utility.getConfig("logName",configFileName);
        // resultCalibrationName=Utility.getConfig("resultCalibrationName",configFileName);

    }

    public static void klPart() throws FileNotFoundException{

        trueKL = Double.parseDouble(Utility.getConfig("trueKL",configFileName));
        String trueBFString = Utility.getConfig("trueBFTable",configFileName);
        String[] bfPairs = trueBFString.split(";");
        trueBFWeightTable = new HashMap<String,Double> ();
        for (int i=0; i<bfPairs.length; i++){
            String onePair = bfPairs[i];
            if (onePair.length()>0 && onePair.contains(",")){
                String[] values = onePair.split(",");
                String nodeName = values[0];
                Double bf = Double.parseDouble(values[1]);
                if (bf <=0.000000001) { bf = 0.000000001; }
                trueBFWeightTable.put(nodeName, bf);
            }
        }

        KL_targetData_learnedSourceModel = Double.parseDouble(Utility.getConfig("KL_targetData_learnedSourceModel",configFileName));
        KL_targetData_trueSourceModel = Double.parseDouble(Utility.getConfig("KL_targetData_trueSourceModel",configFileName));

        String learnedSource_nodeKLTableString = Utility.getConfig("nodeKLTable_targetData_learnedSourceModel",configFileName);
        String trueSource_nodeKLTableString = Utility.getConfig("nodeKLTable_targetData_trueSourceModel",configFileName);

        learnedSource_nodeKLTable =  new HashMap<String,Double>() ;
        String[] klPairs = learnedSource_nodeKLTableString.split(";");
        for (int i=0; i<klPairs.length; i++){
            String onePair = klPairs[i];
            if (onePair.length()>0 && onePair.contains(",")){
                String[] values = onePair.split(",");
                String nodeName = values[0];
                Double kl = Double.parseDouble(values[1]);
                learnedSource_nodeKLTable.put(nodeName, kl);
            }
        }

        trueSource_nodeKLTable =  new HashMap<String,Double>() ;
        klPairs = trueSource_nodeKLTableString.split(";");
        for (int i=0; i<klPairs.length; i++){
            String onePair = klPairs[i];
            if (onePair.length()>0 && onePair.contains(",")){
                String[] values = onePair.split(",");
                String nodeName = values[0];
                Double kl = Double.parseDouble(values[1]);
                trueSource_nodeKLTable.put(nodeName, kl);
            }
        }
    }

    public static void setConfig(String KL) throws FileNotFoundException{

        featureSelectionApproach = Utility.getConfig("featureSelectionApproach",configFileName);
        modelLearningApproach = Utility.getConfig("modelLearningApproach",configFileName);

        //source folder
        sourcePart();

        // target folder
        targetPart();

        // log file
        logPart();

        // If KL is chosen,
        if(KL.equals("on"))
            klPart();
    }


    //------------------------This is just write configuration into file-----------------------------------------//
    // Thus, I create a separate function to place those code//
    // 这个方法里面的code 是记录的，就是写进一个新的文件。保持记录的一个功用，所以放在一个function 里面//
    public static void nothingFunction(String configLoc, String fileName, String KL) throws Exception{
        configFileName = configLoc + fileName + ".txt";
        //configFileName = "/Users/johnsong/Downloads/BNTL-mainpackage-utility-package/BNTL-test/000influenza_ac_topazac_slc_topazslc/utility/folder10_Influenza_CFS_acmodel_0810_topazac_size50_slcmodel_0810_topazslc.bif_targetDataSize_1000-IG.txt";
        //
        setConfig(KL);

        // create a file with
        //
        printoutAUC = new PrintWriter(new File(resultLoc+"auc-"+ fileName.replace(".bif","") + "-all-targetFilter.csv"));
        printoutAUC.println(configFileName);

        // if there is KL method, the log will record, otherwise skip this part.
        if(KL.equals("on")){
            printoutAUC.println("KL_targetData_learnedSourceModel:"+KL_targetData_learnedSourceModel);

            printoutAUC.println("nodeKLTable_targetData_learnedSourceModel:");

            // comment out because of there is no such value

            String[] keyList = learnedSource_nodeKLTable.keySet().toArray(new String[0]);

            for (int i=0; i<keyList.length; i++){
                String key = keyList[i];
                printoutAUC.print(key+","+ learnedSource_nodeKLTable.get(key)+";");
            }

            printoutAUC.println();

            printoutAUC.println("KL_targetData_trueSourceModel:"+KL_targetData_trueSourceModel);
            printoutAUC.println("nodeKLTable_targetData_trueSourceModel:");
            keyList = trueSource_nodeKLTable.keySet().toArray(new String[0]);
            for (int i=0; i<keyList.length; i++){
                String key = keyList[i];
                printoutAUC.print(key+","+ trueSource_nodeKLTable.get(key)+";");
            }
            printoutAUC.println();
        }


    }
//------------------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------------------//
//    public static void runOneExperiment( String experimentName) throws Exception{
//        for (int i=2; i<=2; i++){
//            int folderNum = i;	//"D:/Dropbox/YeEclipseThesis_Laptop/ProcessSimulatedInfluenza_Laptop/folder" + folderNum +
//            String folderExperimentLoc = "./000influenza_" + experimentName  + "/";
//            String configLoc = new String(folderExperimentLoc+"utility/");
//            File folder = new File(configLoc);
//            for (File fileEntry : folder.listFiles()) {
//                if (fileEntry.isFile()) {
//                    String utilityFileName = fileEntry.getName();
//                    System.out.println(utilityFileName);
//                    if (utilityFileName.contains("_targetDataSize_")){
//                        String fileName = utilityFileName.replace(".txt", "");
//                        int sourceSize = Integer.parseInt(fileName.split("_size")[1].split("_")[0]);
//                        int targetSize = Integer.parseInt(fileName.split("targetDataSize_")[1].replace(".txt", "").replace("-IG", ""));
//                        if ((sourceSize==50 | sourceSize==8000) && (targetSize==50 | targetSize==8000)) {
//                            runOneConfig(configLoc,fileName, folderNum);
//                        }
//                    }
//                }
//            }
//        }
//    }


//==========================================================================================================================================================
 //  public void sourceTrue(Instances instances) throws Exception{
        //===== start  source_true_model  &  Target Training Data=========================================
        // source_true_model
        // target training Data
        // trueKL= 1.8224292702088067
        //learnedSource_nodeKLTable (nodeKLTable_targetData_learnedSourceModel)
        // avgKL (never used)  default = 0.0
        // avgBFWeightTable (never get input)
        //KL_targetData_trueSourceModel
        //acmodel_0810_topazac.bif

// 1. KL_targetData_trueSourceModel
//        String startNetworkName = sourceTrueModelName;
//        printoutAUC.println("startNetworkName:" + startNetworkName);
//        BayesNet bn = new BayesNet();
//        //KL_targetData_trueSourceModel=1.660745666413443
//        transferLearningWeight = new String("KL_targetData_trueSourceModel");
//
//        //acmodel_0810_topazac-slcmodel_0810_topazslc_seed1191167696_size1000-priorModelApproach-KL_targetData_trueSourceModel.bif
//        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
//                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
//        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
//                trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable));
//        // build
//        bn.buildClassifier(instances);
//        // Evaluation part
//        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);
//
//// 2. nodeKLTable_targetData_trueSourceModel
//        bn = new BayesNet();
//        transferLearningWeight = new String("nodeKLTable_targetData_trueSourceModel");
//
//        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
//                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
//        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
//                trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable));
//        bn.buildClassifier(instances);
//        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);
//
//        // 3. unadjust=============================================================================================================
//        bn = new BayesNet();
//        transferLearningWeight = new String("unadjust");
//        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
//                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
//        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
//                trueKL,trueBFWeightTable, avgKL,avgBFWeightTable));
//        bn.buildClassifier(instances);
//        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);
//
//        // 4. ratio
//        bn = new BayesNet();
//        transferLearningWeight = new String("ratio");
//        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
//                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
//        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
//                trueKL,trueBFWeightTable, avgKL,avgBFWeightTable));
//        bn.buildClassifier(instances);
//        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);
//
//
//        // 5. trueBayesFactor
//        bn = new BayesNet();
//        transferLearningWeight = new String("trueBayesFactor");
//        targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
//                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
//        bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
//                trueKL,trueBFWeightTable, avgKL,avgBFWeightTable));
//        bn.buildClassifier(instances);
//        printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);


//-----------------------------------------------------------------------------------------------------------//


//    public static List<File> GetFolder(String configLoc ){
//        File directory = new File(configLoc);
//        File [] files = directory.listFiles();
//        List<File> config = new ArrayList<>();
//        for(File f: files){
//            if(f.isFile() && f.getName().endsWith(".txt")){
//                config.add(f);
//                //System.out.println(f);
//            }
//        }
//        return config;
//    }


    //=======================================================================================================//
    /**
     * the most important part of the experiment,
     * core code for transferm learning
     */

    //new BNTL(configFileName,"priorModelApproach", "KL_targetData_trueSourceModel",
    // "acmodel_0810_topazac.bif", trueKL, avgKL,avgBFWeightTable);


    // same structure but different parameters.

    // load multiple models into list
    // In readNetwork() ---> modify path for multi-source models-------|
    //ArrayList<SMILEBayesNet>  netWorkList = readNetwork();     //------|
    //-----------------------------------------------------------------|
//
//            bn.setSearchAlgorithm(new BNTL_TEST_MODIFY(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
//                  trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable,netWorkList));
//=======================================================================================================//




    // 跑实验的路径2
    //String folderExperimentLoc = "./000.experiment_SLC2AC/";
    //String fileName ="FLU-SLC2AC-0809-transfer20090118-IG001-NB";

    //    public static void setConfig() throws FileNotFoundException{
//        sourceModelLoc=Utility.getConfig("sourceModelLoc",configFileName);
//
//        sourceTrueModelName=Utility.getConfig("sourceTrueModelName",configFileName);
//        sourceCleanInjectModelXML=Utility.getConfig("sourceCleanInjectModelXML",configFileName);
//
//        // multiple version to get list of network
//        // use [1.xdsl, 2.xdsl, 3.xdsl....]
//        // add by John Song for multi-version
//        String temp =Utility.getConfig("sourceCleanInjectModelXDSL",configFileName);
//        String [] temp1 = temp.split(",");
//        sourceClearnInjectModelXDSL= new ArrayList<>();
//        for(String a: temp1){
//            sourceClearnInjectModelXDSL.add(a);
//        }
//
//
//        sourceSimulateDataLoc=Utility.getConfig("sourceSimulateDataLoc",configFileName);
//        sourceSimulateDataName=Utility.getConfig("sourceSimulateDataName",configFileName);
//        sourceDataLoc=Utility.getConfig("sourceDataLoc",configFileName);
//        sourceDataName=Utility.getConfig("sourceDataName",configFileName);
//        sourceDataSize=Utility.getConfig("sourceDataSize",configFileName);
//        targetModelLoc=Utility.getConfig("targetModelLoc",configFileName);
//        targetTrueModelName=Utility.getConfig("targetTrueModelName",configFileName);
//
//
//        targetDataLoc=Utility.getConfig("targetDataLoc",configFileName);
//        targetDataName=Utility.getConfig("targetDataName",configFileName);
//        targetNodeName=Utility.getConfig("targetNodeName",configFileName);
//
//        targetTestDataName=Utility.getConfig("targetTestDataName",configFileName);
//
//
//        resultLoc=Utility.getConfig("resultLoc",configFileName);
//        resultProbName=Utility.getConfig("resultProbName",configFileName);
//        resultAUCName=Utility.getConfig("resultAUCName",configFileName);
//        resultCalibrationName=Utility.getConfig("resultCalibrationName",configFileName);
//
//
//        utilityLoc=Utility.getConfig("utilityLoc",configFileName);
//        temporaryFileName=Utility.getConfig("temporaryFileName",configFileName);
//        hashCodeName=Utility.getConfig("hashCodeName",configFileName);
//
//        logLoc=Utility.getConfig("logLoc",configFileName);
//        logName=Utility.getConfig("logName",configFileName);
//        targetClassName = Utility.getConfig("targetClassName",configFileName);
//
//        // targetClassname = ""
//        if (targetClassName==null){
//            targetClassName = new String("class");
//        }
//
//
//
//        featureSelectionApproach = Utility.getConfig("featureSelectionApproach",configFileName);
//
//        modelLearningApproach = Utility.getConfig("modelLearningApproach",configFileName);
//
//
//        sourceLearnedModelName=sourceDataName.replace(".arff", "-" + featureSelectionApproach + "-" + modelLearningApproach + ".bif");
//        targetLearnedModelName=targetDataName.replace(".arff", "-" + featureSelectionApproach + "-" + modelLearningApproach + ".bif");
//        sourceLearnedCleanedModelName = sourceLearnedModelName.replace(".bif", "")+"_cleaned.bif";
//        sourceTrueCleanedModelName = sourceTrueModelName.replace(".bif", "")+"_cleaned.bif";
//
//
//        // add if(){} statement to avoid calculation -- John Song
//        if(modelLearningApproach.equals("K2")){
//
//            trueKL = Double.parseDouble(Utility.getConfig("trueKL",configFileName));
//
//
//            String trueBFString = Utility.getConfig("trueBFTable",configFileName);
//            String[] bfPairs = trueBFString.split(";");
//            trueBFWeightTable = new HashMap<String,Double> ();
//            for (int i=0; i<bfPairs.length; i++){
//                String onePair = bfPairs[i];
//                if (onePair.length()>0 && onePair.contains(",")){
//                    String[] values = onePair.split(",");
//                    String nodeName = values[0];
//                    Double bf = Double.parseDouble(values[1]);
//                    if (bf <=0.000000001) { bf = 0.000000001; }
//                    trueBFWeightTable.put(nodeName, bf);
//                }
//            }
//
//            KL_targetData_learnedSourceModel = Double.parseDouble(Utility.getConfig("KL_targetData_learnedSourceModel",configFileName));
//            KL_targetData_trueSourceModel = Double.parseDouble(Utility.getConfig("KL_targetData_trueSourceModel",configFileName));
//
//            String learnedSource_nodeKLTableString = Utility.getConfig("nodeKLTable_targetData_learnedSourceModel",configFileName);
//            String trueSource_nodeKLTableString = Utility.getConfig("nodeKLTable_targetData_trueSourceModel",configFileName);
//
//            learnedSource_nodeKLTable =  new HashMap<String,Double>() ;
//            String[] klPairs = learnedSource_nodeKLTableString.split(";");
//            for (int i=0; i<klPairs.length; i++){
//                String onePair = klPairs[i];
//                if (onePair.length()>0 && onePair.contains(",")){
//                    String[] values = onePair.split(",");
//                    String nodeName = values[0];
//                    Double kl = Double.parseDouble(values[1]);
//                    learnedSource_nodeKLTable.put(nodeName, kl);
//                }
//            }
//
//            trueSource_nodeKLTable =  new HashMap<String,Double>() ;
//            klPairs = trueSource_nodeKLTableString.split(";");
//            for (int i=0; i<klPairs.length; i++){
//                String onePair = klPairs[i];
//                if (onePair.length()>0 && onePair.contains(",")){
//                    String[] values = onePair.split(",");
//                    String nodeName = values[0];
//                    Double kl = Double.parseDouble(values[1]);
//                    trueSource_nodeKLTable.put(nodeName, kl);
//                }
//            }
//        }
//
//
//
//    }



}




