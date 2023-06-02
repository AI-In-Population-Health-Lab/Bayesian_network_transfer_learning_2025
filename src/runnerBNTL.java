
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

// weka package
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



public class runnerBNTL {

    static String configFileName;
    static String sourceModelLoc,sourceLearnedModelName,sourceLearnedCleanedModelName, sourceTrueModelName,sourceTrueCleanedModelName, sourceCleanInjectModelXML,
            sourceClearnInjectModelXDSL,sourceSimulateDataLoc,sourceSimulateDataName,sourceDataLoc,sourceDataName,sourceDataSize,
            targetModelLoc,targetTrueModelName,targetLearnedModelName,targetFinalModelName, targetDataLoc,targetDataName,targetNodeName,
            targetTestDataName,resultLoc,resultProbName,resultAUCName,resultCalibrationName,
            utilityLoc,temporaryFileName,hashCodeName,logLoc,logName, targetClassName;

    static String featureSelectionApproach, modelLearningApproach;
    static String transferLearningApproach, transferLearningWeight;
    static PrintWriter printoutAUC;
    static double trueKL, avgKL;
    static HashMap<String,Double> trueBFWeightTable, avgBFWeightTable;

    static double KL_targetData_learnedSourceModel, KL_targetData_trueSourceModel;
    static HashMap<String,Double> learnedSource_nodeKLTable;
    static HashMap<String,Double> trueSource_nodeKLTable;

    public static void main(String[] args) throws Exception {
        // TODO Auto-generated method stub
//
//		String path = "/Users/johnsong/Documents/ye_lab/bayesnet_tranforming/lib/jsmile.jnilib";
//		System.setProperty("jsmile.native.library", path);

        new smile.License(
                "SMILE LICENSE d918bddd 871cb16c 3d01a731 " +
                        "THIS IS AN ACADEMIC LICENSE AND CAN BE USED " +
                        "SOLELY FOR ACADEMIC RESEARCH AND TEACHING, " +
                        "AS DEFINED IN THE BAYESFUSION ACADEMIC " +
                        "SOFTWARE LICENSING AGREEMENT. " +
                        "Serial #: 75ivcaomfx4xfthy077ica6bx " +
                        "Issued for: Ye Ye (yeyewy@gmail.com) " +
                        "Academic institution: University of Pittsburgh " +
                        "Valid until: 2023-09-29 " +
                        "Issued by BayesFusion activation server",
                new byte[] {
                        103,99,55,-38,-42,24,-11,-31,17,-23,-92,-121,-45,-108,-70,90,
                        -123,-7,91,-2,-11,-51,-108,-44,-35,33,93,-11,-53,-72,31,-15,
                        23,43,-31,-65,81,101,47,-123,95,-40,-4,-101,90,98,-101,-11,
                        -27,49,-121,-38,-92,-97,73,39,-70,-40,95,-21,13,-111,-83,-98
                }
        );

        System.out.println("hello");
        String folderExperimentLoc = "./000influenza_ac_topazac_slc_topazslc/";
        String configLoc = new String(folderExperimentLoc+"utility/");
        String fileName = "folder10_Influenza_CFS_acmodel_0810_topazac_size50_slcmodel_0810_topazslc.bif_targetDataSize_1000-IG";

        runOneConfig(configLoc,fileName, 10);

        //runOneExperiment("ac_topazac_slc_topazslc");

    }
    public static void runOneExperiment( String experimentName) throws Exception{
        for (int i=2; i<=2; i++){
            int folderNum = i;	//"D:/Dropbox/YeEclipseThesis_Laptop/ProcessSimulatedInfluenza_Laptop/folder" + folderNum +
            String folderExperimentLoc = "./000influenza_" + experimentName  + "/";
            String configLoc = new String(folderExperimentLoc+"utility/");
            File folder = new File(configLoc);
            for (File fileEntry : folder.listFiles()) {
                if (fileEntry.isFile()) {
                    String utilityFileName = fileEntry.getName();
                    System.out.println(utilityFileName);
                    if (utilityFileName.contains("_targetDataSize_")){
                        String fileName = utilityFileName.replace(".txt", "");
                        int sourceSize = Integer.parseInt(fileName.split("_size")[1].split("_")[0]);
                        int targetSize = Integer.parseInt(fileName.split("targetDataSize_")[1].replace(".txt", "").replace("-IG", ""));
                        if ((sourceSize==50 | sourceSize==8000) && (targetSize==50 | targetSize==8000)) {
                            runOneConfig(configLoc,fileName, folderNum);
                        }
                    }
                }
            }
        }
    }


    public static void runOneConfig(String configLoc, String fileName, int folderNum) throws Exception{
        configFileName = configLoc + fileName + ".txt";
        //configFileName = "/Users/johnsong/Downloads/BNTL-mainpackage-utility-package/BNTL-test/000influenza_ac_topazac_slc_topazslc/utility/folder10_Influenza_CFS_acmodel_0810_topazac_size50_slcmodel_0810_topazslc.bif_targetDataSize_1000-IG.txt";

        setConfig();

        printoutAUC = new PrintWriter(new File(resultLoc+"auc-"+ fileName.replace(".bif","") + "-all-targetFilter.csv"));
        printoutAUC.println(configFileName);

        printoutAUC.println("KL_targetData_learnedSourceModel:"+KL_targetData_learnedSourceModel);

        printoutAUC.println("nodeKLTable_targetData_learnedSourceModel:");
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
        printBaseLinePerformance();
        getModels(configFileName);
        printoutAUC.close();
    }

    public static void getModels(String configFileName) throws Exception{
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
            instances = filtedData;

            System.out.println("Source model is: " + sourceLearnedModelName);

            System.out.println("Target training data is: " + targetDataName);

            System.out.println("===================Transformation ==========================");

            transferLearningApproach = new String("priorModelApproach");

            //start search from true source model instead

            String startNetworkName = sourceTrueModelName;
            printoutAUC.println("startNetworkName:" + startNetworkName);

            BayesNet bn = new BayesNet();

            transferLearningWeight = new String("KL_targetData_trueSourceModel");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);


            bn = new BayesNet();
            transferLearningWeight = new String("nodeKLTable_targetData_trueSourceModel");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);


            bn = new BayesNet();
            transferLearningWeight = new String("unadjust");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,trueBFWeightTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);

            bn = new BayesNet();
            transferLearningWeight = new String("ratio");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,trueBFWeightTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);


            bn = new BayesNet();
            transferLearningWeight = new String("trueBayesFactor");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,trueBFWeightTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);

            // start search from learned source model
            startNetworkName = sourceLearnedModelName;
            printoutAUC.println("startNetworkName:" + startNetworkName);

            bn = new BayesNet();
            transferLearningWeight = new String("KL_targetData_learnedSourceModel");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);

            bn = new BayesNet();
            transferLearningWeight = new String("nodeKLTable_targetData_learnedSourceModel");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,learnedSource_nodeKLTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);


            bn = new BayesNet();
            transferLearningWeight = new String("unadjust");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,trueBFWeightTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);

            bn = new BayesNet();
            transferLearningWeight = new String("ratio");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,trueBFWeightTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);


            bn = new BayesNet();
            transferLearningWeight = new String("trueBayesFactor");
            targetFinalModelName=startNetworkName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                    + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";
            bn.setSearchAlgorithm(new BNTL(configFileName,transferLearningApproach,transferLearningWeight, startNetworkName,
                    trueKL,trueBFWeightTable, avgKL,avgBFWeightTable));
            bn.buildClassifier(instances);
            printOnePerformance(targetModelLoc,targetFinalModelName, targetDataLoc+targetTestDataName);

        } catch (IOException e) {
            e.printStackTrace();
        }
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
        System.out.println("------Cleaning source specific features");
        PrintWriter printout = new PrintWriter(new File(utilityLoc+temporaryFileName));
        printout.println(bayesNet.graph());
        printout.flush(); printout.close();
        BIFReader reader = new BIFReader();
        EditableBayesNet tempNet = new EditableBayesNet(reader.processFile(utilityLoc+temporaryFileName));
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
        String testDataset = new String(targetDataLoc+targetTestDataName);

        String trueTargetModel = new String(targetModelLoc+targetTrueModelName);
        BIFReader ture_target_BayesNet = new BIFReader();
        ture_target_BayesNet.processFile(trueTargetModel);
        System.out.println("AUC_true_target:" + evaluateBN(targetTrueModelName, ture_target_BayesNet, testDataset));

        String targetOnlyModel = new String(targetModelLoc+targetLearnedModelName);
        BIFReader targetOnlyModel_BayesNet = new BIFReader();
        targetOnlyModel_BayesNet.processFile(targetOnlyModel);
        System.out.println("AUC_target_only:" + evaluateBN(targetLearnedModelName, targetOnlyModel_BayesNet, testDataset));

        cleanSourceTrueModel();
        String trueSourceModel = new String(sourceModelLoc+sourceTrueCleanedModelName);
        BIFReader ture_source_BayesNet = new BIFReader();
        ture_source_BayesNet.processFile(trueSourceModel);
        System.out.println("AUC_true_source_clean:" + evaluateBN(sourceTrueCleanedModelName, ture_source_BayesNet, testDataset));

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

    public static void setConfig() throws FileNotFoundException{
        sourceModelLoc=Utility.getConfig("sourceModelLoc",configFileName);

        sourceTrueModelName=Utility.getConfig("sourceTrueModelName",configFileName);
        sourceCleanInjectModelXML=Utility.getConfig("sourceCleanInjectModelXML",configFileName);
        sourceClearnInjectModelXDSL=Utility.getConfig("sourceClearnInjectModelXDSL",configFileName);
        sourceSimulateDataLoc=Utility.getConfig("sourceSimulateDataLoc",configFileName);
        sourceSimulateDataName=Utility.getConfig("sourceSimulateDataName",configFileName);
        sourceDataLoc=Utility.getConfig("sourceDataLoc",configFileName);
        sourceDataName=Utility.getConfig("sourceDataName",configFileName);
        sourceDataSize=Utility.getConfig("sourceDataSize",configFileName);
        targetModelLoc=Utility.getConfig("targetModelLoc",configFileName);
        targetTrueModelName=Utility.getConfig("targetTrueModelName",configFileName);


        targetDataLoc=Utility.getConfig("targetDataLoc",configFileName);
        targetDataName=Utility.getConfig("targetDataName",configFileName);
        targetNodeName=Utility.getConfig("targetNodeName",configFileName);
        targetTestDataName=Utility.getConfig("targetTestDataName",configFileName);
        resultLoc=Utility.getConfig("resultLoc",configFileName);
        resultProbName=Utility.getConfig("resultProbName",configFileName);
        resultAUCName=Utility.getConfig("resultAUCName",configFileName);
        resultCalibrationName=Utility.getConfig("resultCalibrationName",configFileName);
        utilityLoc=Utility.getConfig("utilityLoc",configFileName);
        temporaryFileName=Utility.getConfig("temporaryFileName",configFileName);
        hashCodeName=Utility.getConfig("hashCodeName",configFileName);
        logLoc=Utility.getConfig("logLoc",configFileName);
        logName=Utility.getConfig("logName",configFileName);
        targetClassName = Utility.getConfig("targetClassName",configFileName);
        if (targetClassName==null){
            targetClassName = new String("class");
        }

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

        featureSelectionApproach = Utility.getConfig("featureSelectionApproach",configFileName);
        modelLearningApproach = Utility.getConfig("modelLearningApproach",configFileName);

        sourceLearnedModelName=sourceDataName.replace(".arff", "-" + featureSelectionApproach + "-" + modelLearningApproach + ".bif");
        targetLearnedModelName=targetDataName.replace(".arff", "-" + featureSelectionApproach + "-" + modelLearningApproach + ".bif");
        sourceLearnedCleanedModelName = sourceLearnedModelName.replace(".bif", "")+"_cleaned.bif";
        sourceTrueCleanedModelName = sourceTrueModelName.replace(".bif", "")+"_cleaned.bif";

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






}

