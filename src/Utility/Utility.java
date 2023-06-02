package Utility;

import java.io.File;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

// math3
import org.apache.commons.math3.special.Gamma;

// smile and weka
import smile.Network;
import smile.SMILEException;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.bayes.net.BayesNetGenerator;
import weka.classifiers.bayes.net.ParentSet;
import weka.classifiers.bayes.net.estimate.DiscreteEstimatorBayes;
import weka.core.*;
import weka.core.converters.ArffLoader;

// own package
import edu.pitt.dbmi.dataset.ConvertBIFtoGENIE;
import edu.pitt.isp.sverchkov.exec.CompositeException;
import edu.pitt.isp.sverchkov.smile.SMILEBayesNet;


public class Utility {

    static int numberOfSimulation = 100;
    static int numberOfSimulation_Estimation = 1000;

    public static void main(String args[]) throws Exception {

    }







//	public static double getJointProb(BayesNet BN, int iAttribute, int iAttributeConfig, int iParentConfig){
//		double logP = 0;
//		ArrayList<Integer> attributeList = new ArrayList<Integer>();
//		logP = logP + BN.m_Distributions[iAttribute][iParentConfig].getProbability(iAttributeConfig);
//		attributeList.add(iAttribute);
//		for (int iParent = 0; iParent < BN.getParentSet(iAttribute).getNrOfParents(); iParent++) {
//            int nParent = BN.getParentSet(iAttribute).getParent(iParent);
//            attributeList.add(nParent);
//		}
//		BN.getParentSets;
//
//		return null;
//	}
//
//	public static double getJointProb(BayesNet BN, int iAttribute, int iAttributeConfig, int iParentConfig){
//		double logP = 0;
//		ArrayList<Integer> attributeList = new ArrayList<Integer>();
//		logP = logP + BN.m_Distributions[iAttribute][iParentConfig].getProbability(iAttributeConfig);
//		attributeList.add(iAttribute);
//		for (int iParent = 0; iParent < BN.getParentSet(iAttribute).getNrOfParents(); iParent++) {
//            int nParent = BN.getParentSet(iAttribute).getParent(iParent);
//            attributeList.add(nParent);
//		}
//		BN.getParentSets;
//
//		return null;
//	}

    public static double getTrueKL(HashMap<String,Double> klTable){
        String[] keys = klTable.keySet().toArray(new String[0]);
        Double kl = 0.0;
        for (int i=0; i<keys.length; i++){
            kl = kl +  klTable.get(keys[i]);
        }
        return kl;
    }

//	public static double getTrueKL(String sourceModelLoc, String sourceModelName,
//			String targetModelLoc, String targetModelName){
//
//		Double kl = 0.0;
//		return kl;
//	}
//





    /**
     * Return a weight table <NodeName,Bayes Factor as a WeightForSourceData>
     * When P(Du|Mu) >= P(Ds|Ms)*P(Dt|Mt). The node is identical in source and target. So,
     * 		assign 1 for each source instance.
     * When P(Du|Mu) < P(Ds|Ms)*P(Dt|Mt). Assign weight = 1/BayesFactor = P(Du|Mu)/[P(Ds|Ms)*P(Dt|Mt)]
     * @param targetNodeName
     * @param sharedNodeList
     * @param sourceData
     * @param targetData
     * @return
     * @throws Exception
     */
    public static HashMap<String,Double> getTrueAverageBayesFactorTable(String targetNodeName,
                                                                        String sourceModelLoc, String sourceModelName, String targetModelLoc, String targetModelName,
                                                                        String generatedDataLoc) throws Exception{
        //Generate source data and calculate KL 10 times
        BayesNet source_BayesNet = new BIFReader();
        ((BIFReader) source_BayesNet).processFile(sourceModelLoc+sourceModelName);
        BayesNet target_BayesNet = new BIFReader();
        ((BIFReader) target_BayesNet).processFile(targetModelLoc+targetModelName);
        ArrayList<String> sharedNodeList = new ArrayList<String>();
        ArrayList<String> sourceNodeList = new ArrayList<String>();
        for (int i=0; i<source_BayesNet.getNrOfNodes(); i++){
            String oneNode = source_BayesNet.getNodeName(i);
            sourceNodeList.add(oneNode);
        }
        for (int i=0; i<target_BayesNet.getNrOfNodes(); i++){
            String oneNode = target_BayesNet.getNodeName(i);
            if (sourceNodeList.contains(oneNode)){
                sharedNodeList.add(oneNode);
            }
        }
        HashMap<String,Double> averageWeightTable = new HashMap<String,Double> ();
        for (int iNode=0; iNode<sharedNodeList.size(); iNode++){
            String iNodeName = sharedNodeList.get(iNode);
            averageWeightTable.put(iNodeName, new Double("0.0"));
        }

        for (int round=1; round<=numberOfSimulation; round++){
            String sampleSize = "10000";
            Random seedGenerator = new Random();
            String seed = seedGenerator.nextInt()+"";
            sourceModelName = sourceModelName.replace(".bif","");
            String modelName = new String(sourceModelName);
            String generatedSourceData = generatedDataLoc + "tempForBayesFactor.arff";
            generateData(sampleSize, seed, sourceModelLoc, modelName, generatedDataLoc, "tempForBayesFactor");
            ArffLoader arff = new ArffLoader();
            arff.setFile(new File(generatedSourceData));
            Instances  sourceData = arff.getDataSet();


            seed = seedGenerator.nextInt()+"";
            targetModelName = targetModelName.replace(".bif","");
            modelName = new String(targetModelName);
            String generatedTargetData = generatedDataLoc + "tempForTargetBayesFactor.arff";
            generateData(sampleSize, seed, targetModelLoc, modelName, generatedDataLoc, "tempForTargetBayesFactor");
            ArffLoader arff2 = new ArffLoader();
            arff2.setFile(new File(generatedTargetData));
            Instances  targetData = arff2.getDataSet();

            HashMap<String,Double> currentWeightTable = getBayesFactorTable(targetNodeName, sharedNodeList,
                    sourceData, targetData);

            for (int iNode=0; iNode<sharedNodeList.size(); iNode++){
                String iNodeName = sharedNodeList.get(iNode);
                double currentWeight = currentWeightTable.get(iNodeName);
                double currentAverageWeight = averageWeightTable.get(iNodeName);
                averageWeightTable.put(iNodeName,currentAverageWeight+currentWeight);
                //System.out.print(iNodeName+":"+currentWeight + ",");
            }
            //System.out.println();
        }//end round
        System.out.println("printing average Bayes factor: ");
        for (int iNode=0; iNode<sharedNodeList.size(); iNode++){
            String iNodeName = sharedNodeList.get(iNode);
            double averageWeight = 1.0*averageWeightTable.get(iNodeName)/numberOfSimulation;
            if(averageWeight <=0.000000001) {
                averageWeight = 0.000000001;
            }
            averageWeightTable.put(iNodeName,averageWeight);
            System.out.println(iNodeName+":"+ averageWeight);
        }
        return averageWeightTable;
    }



    /**
     * Return a weight table <NodeName,Bayes Factor as a WeightForSourceData>
     * When P(Du|Mu) >= P(Ds|Ms)*P(Dt|Mt). The node is identical in source and target. So,
     * 		assign 1 for each source instance.
     * When P(Du|Mu) < P(Ds|Ms)*P(Dt|Mt). Assign weight = 1/BayesFactor = P(Du|Mu)/[P(Ds|Ms)*P(Dt|Mt)]
     * @param targetNodeName
     * @param sharedNodeList
     * @param sourceData
     * @param targetData
     * @return
     * @throws Exception
     */
    public static HashMap<String,Double> getAverageBayesFactorTable(String targetNodeName,
                                                                    String targetDataLoc, String targetDataName,
                                                                    String sourceSize, String sourceModelLoc, String sourceModelName,
                                                                    String generatedDataLoc, String sourceDataLoc, String sourceDataName) throws Exception{

        ArffLoader arff1 = new ArffLoader();
        arff1.setFile(new File(targetDataLoc+targetDataName));
        Instances  targetData = arff1.getDataSet();

        //Generate source data and calculate KL 10 times
        BayesNet source_BayesNet = new BIFReader();
        ((BIFReader) source_BayesNet).processFile(sourceModelLoc+sourceModelName);
        ArrayList<String> sharedNodeList = new ArrayList<String>();
        ArrayList<String> sourceNodeList = new ArrayList<String>();
        for (int i=0; i<source_BayesNet.getNrOfNodes(); i++){
            String oneNode = source_BayesNet.getNodeName(i);
            sourceNodeList.add(oneNode);
        }
        for (int i=0; i<targetData.numAttributes(); i++){
            String oneNode = targetData.attribute(i).name();
            if (sourceNodeList.contains(oneNode)){
                sharedNodeList.add(oneNode);
            }
        }
        HashMap<String,Double> averageWeightTable = new HashMap<String,Double> ();
        for (int iNode=0; iNode<sharedNodeList.size(); iNode++){
            String iNodeName = sharedNodeList.get(iNode);
            averageWeightTable.put(iNodeName, new Double("0.0"));
        }
        for (int round=1; round<=numberOfSimulation_Estimation; round++){
            String sampleSize = sourceSize;
            Random seedGenerator = new Random();
            String seed = seedGenerator.nextInt()+"";
            sourceModelName = sourceModelName.replace(".bif","");
            String modelName = new String(sourceModelName);
            String generatedSourceData = generatedDataLoc + "tempForBayesFactor.arff";
            generateData(sampleSize, seed, sourceModelLoc, modelName, generatedDataLoc, "tempForBayesFactor");
            ArffLoader arff = new ArffLoader();
            arff.setFile(new File(generatedSourceData));
            Instances  sourceData = arff.getDataSet();
            HashMap<String,Double> currentWeightTable = getBayesFactorTable(targetNodeName, sharedNodeList,
                    sourceData, targetData);
//			for (int iNode=0; iNode<sharedNodeList.size(); iNode++){
//				String iNodeName = sharedNodeList.get(iNode);
//				System.out.println(iNodeName);
//			}

            for (int iNode=0; iNode<sharedNodeList.size(); iNode++){
                String iNodeName = sharedNodeList.get(iNode);
                double currentWeight = currentWeightTable.get(iNodeName);
                double currentAverageWeight = averageWeightTable.get(iNodeName);
                averageWeightTable.put(iNodeName,currentAverageWeight+currentWeight);
                //System.out.print(iNodeName+":"+currentWeight + ",");
            }
            //System.out.println();
        }//end round
        System.out.println("printing average Bayes factor: ");
        for (int iNode=0; iNode<sharedNodeList.size(); iNode++){
            String iNodeName = sharedNodeList.get(iNode);
            double averageWeight = 1.0*averageWeightTable.get(iNodeName)/numberOfSimulation_Estimation;
            if(averageWeight <=0.000000001) {
                averageWeight = 0.000000001;
            }
            averageWeightTable.put(iNodeName,averageWeight);
            System.out.println(iNodeName+":"+ averageWeight);
        }
//		//get true
//		System.out.println("printing true Bayes factor: ");
//		ArffLoader arff2 = new ArffLoader();
//		arff2.setFile(new File(sourceDataLoc+sourceDataName));
//		Instances  sourceData2 = arff2.getDataSet();
//		HashMap<String,Double> trueWeightTable = getBayesFactorTable(targetNodeName, sharedNodeList,
//				sourceData2, targetData);
//		for (int iNode=0; iNode<sharedNodeList.size(); iNode++){
//			String iNodeName = sharedNodeList.get(iNode);
//			System.out.println(iNodeName+":"+ trueWeightTable.get(iNodeName));
//		}

        return averageWeightTable;
    }


    /**
     * Return a weight table <NodeName,Bayes Factor as a WeightForSourceData>
     * When P(Du|Mu) >= P(Ds|Ms)*P(Dt|Mt). The node is identical in source and target. So,
     * 		assign 1 for each source instance.
     * When P(Du|Mu) < P(Ds|Ms)*P(Dt|Mt). Assign weight = 1/BayesFactor = P(Du|Mu)/[P(Ds|Ms)*P(Dt|Mt)]
     * @param targetNodeName
     * @param sharedNodeList
     * @param sourceData
     * @param targetData
     * @return
     */
    public static HashMap<String,Double> getBayesFactorTable(String targetNodeName, ArrayList<String> sharedNodeList,
                                                             Instances sourceData, Instances targetData){
        HashMap<String,Integer> nameIndexTableSource = new HashMap<String,Integer>();
        for (int i=0; i<sourceData.numAttributes(); i++){
            String attName = sourceData.attribute(i).name();
            nameIndexTableSource.put(attName, i);
        }
        HashMap<String,Integer> nameIndexTableTarget = new HashMap<String,Integer>();
        for (int j=0; j<targetData.numAttributes(); j++){
            String attName = targetData.attribute(j).name();
            nameIndexTableTarget.put(attName, j);
        }
        int indexTargetSource = nameIndexTableSource.get(targetNodeName);
        int indexTargetTarget = nameIndexTableTarget.get(targetNodeName);
        ArrayList<String> targetNodeValueList = new ArrayList<String>();
        Attribute targetSource = sourceData.attribute(indexTargetSource);
        for (int i=0; i<targetSource.numValues(); i++){
            String value = targetSource.value(i);
            if (!targetNodeValueList.contains(value)) { targetNodeValueList.add(value); }
        }
        Attribute targetTarget = targetData.attribute(indexTargetTarget);
        for (int j=0; j<targetTarget.numValues(); j++){
            String value = targetTarget.value(j);
            if (!targetNodeValueList.contains(value)) { targetNodeValueList.add(value); }
        }

        HashMap<String,Double> weightTable = new HashMap<String,Double>();
        for (int i=0; i<sharedNodeList.size();i++){
            String nodeName = sharedNodeList.get(i);
            if (nameIndexTableSource.containsKey(nodeName) && nameIndexTableTarget.containsKey(nodeName)) {
                double bayesFactor = getOneBayesFactor(nodeName, targetNodeName, targetNodeValueList,
                        sourceData, nameIndexTableSource,
                        targetData, nameIndexTableTarget);
                //	double weight = Math.pow(2, -bayesFactor); //bayesFactor measures dissimilarity
                double weight = 1.0;
                if (bayesFactor < 1) { weight = bayesFactor ; }
                weightTable.put(nodeName,weight);
            }
        }
        return weightTable;
    }



    /**
     * This one is get BayesFactor for a node, assuming Naive Bayes structure,i.e.,
     * each node is a child of the class node.
     * @param targetNodeName
     * @param nodeName
     * @param sourceData
     * @param targetData
     * @return
     */
    public static double getOneBayesFactor(String nodeName, String targetNodeName, ArrayList<String> targetNodeValueList,
                                           Instances sourceData, HashMap<String,Integer> nameIndexTableSource,
                                           Instances targetData, HashMap<String,Integer> nameIndexTableTarget){
        double logBayesFactor = 0.0;
        ArrayList<String> nodeValueList = new ArrayList<String>(); //all distinct values for the node
        int nodeIndexSource = nameIndexTableSource.get(nodeName);
        int nodeIndexTarget = nameIndexTableTarget.get(nodeName);
        Attribute nodeSource = sourceData.attribute(nodeIndexSource);
        Attribute nodeTarget = targetData.attribute(nodeIndexTarget);
        for (int v=0; v<nodeSource.numValues(); v++){
            String oneValue = nodeSource.value(v);
            if (!nodeValueList.contains(oneValue)) { nodeValueList.add(oneValue); }
        }
        for (int v=0; v<nodeTarget.numValues(); v++){
            String oneValue = nodeTarget.value(v);
            if (!nodeValueList.contains(oneValue)) { nodeValueList.add(oneValue); }
        }

        int q = 1; //number of parent configurations
        //if the node is not class node. The node is a child of the class node.
        //q is number of distinct values of class node.
        if (!nodeName.equals(targetNodeName)) {q = targetNodeValueList.size(); }
        int r = nodeValueList.size();
        //System.out.println(nodeName+":");
        for (int j=1; j<=q; j++){
            String targetNodeValueJ = targetNodeValueList.get(j-1); //targetNode's jth value
            //	System.out.println("target="+targetNodeValueJ);
            HashMap<String,String> assignmentTable = new HashMap<String,String> ();
            if (q>1) {
                assignmentTable.put(targetNodeName, targetNodeValueJ);
            }
            int Dsj = 0;
            int Dtj = 0;
            for (int k=1; k<=r; k++){
                String nodeValueK = nodeValueList.get(k-1) ;
                assignmentTable.put(nodeName, nodeValueK);
                int Dsjk = getCount(assignmentTable,sourceData,nameIndexTableSource);
                int Dtjk = getCount(assignmentTable,targetData,nameIndexTableTarget);
                int Dujk = Dsjk + Dtjk;
                logBayesFactor = logBayesFactor + Gamma.logGamma(1.0*(1+Dujk)) + Gamma.logGamma(1.0)
                        - Gamma.logGamma(1.0*(1+Dsjk)) - Gamma.logGamma(1.0*(1+Dtjk));

                //	System.out.println(nodeName+"="+nodeValueK+","+Dsjk+","+Dtjk+","+Dujk);
                //	System.out.println(Gamma.logGamma(1.0*(1+Dsjk))+","+Gamma.logGamma(1.0*(1+Dtjk))+","+Gamma.logGamma(1.0*(1+Dujk)));
                Dsj = Dsj + Dsjk;
                Dtj = Dtj + Dtjk;
            }

            int Duj = Dsj+Dtj;
            //	System.out.println("  "+Dsj+","+Dtj+","+Duj);
            logBayesFactor = logBayesFactor + Gamma.logGamma(1.0*(r+Dsj)) + Gamma.logGamma(1.0*(r+Dtj))
                    - Gamma.logGamma(1.0*(r+Duj)) - Gamma.logGamma(1.0*(r));
            //	System.out.println(Gamma.logGamma(1.0*(r+Dsj))+","+Gamma.logGamma(1.0*(r+Dtj))+","+Gamma.logGamma(1.0*(r+Duj))
            //			+","+ Gamma.logGamma(1.0*(r)));
        }
        double BayesFactor = Math.exp(logBayesFactor);
        System.out.println(nodeName+","+logBayesFactor+","+BayesFactor);
        return BayesFactor;
    }

    /**
     * Get count for an assignment
     * @param assignmentTable
     * @param data
     * @param nameIndexTable
     * @return
     */
    public static int getCount(HashMap<String,String> assignmentTable, Instances data, HashMap<String,Integer> nameIndexTable){
        int count = 0 ;
        for (int i=0; i<data.numInstances(); i++){
            Instance oneRecord = data.instance(i);
            Boolean allTrue = true;
            String[] nodeList = assignmentTable.keySet().toArray(new String[0]);
            for (int j=0; j<nodeList.length; j++){
                String oneNode = nodeList[j];
                String oneNodeValue = assignmentTable.get(oneNode);
                int index = nameIndexTable.get(oneNode);
                String valueInRecord = oneRecord.stringValue(index);
                if (!oneNodeValue.equals(valueInRecord)) {
                    allTrue = false;
                    break;
                }
            }
            if (allTrue){
                count++;
            }
        }
        return count;
    }




    //Utility.getAverageKL(targetDataLoc+targetDataName,sourceDataSize,sourceModelLoc,sourceLearnedModelName,sourceSimulateDataLoc)
    public static double getTrueAverageKL(String sourceModelLoc, String sourceModelName,
                                          String targetModelLoc, String targetModelName, String generatedDataLoc)
            throws Exception{
        double averageKL = 0.0;
        //Generate source data and calculate KL 10 times

        for (int round=1; round<=numberOfSimulation; round++){
            String sampleSize = "10000";
            Random seedGenerator = new Random();
            String seed = seedGenerator.nextInt()+"";
            sourceModelName = sourceModelName.replace(".bif","");
            String modelName = new String(sourceModelName);
            String generatedSourceData = generatedDataLoc+ "tempSourceForKL.arff";
            generateData(sampleSize, seed, sourceModelLoc, modelName, generatedDataLoc, "tempSourceForKL");
            ArffLoader arff2 = new ArffLoader();
            arff2.setFile(new File(generatedSourceData));
            Instances  sourceData = arff2.getDataSet();

            seed = seedGenerator.nextInt()+"";
            targetModelName = targetModelName.replace(".bif","");
            modelName = new String(targetModelName);
            String generatedTargetData = generatedDataLoc+ "tempTargetForKL.arff";
            generateData(sampleSize, seed, targetModelLoc, modelName, generatedDataLoc, "tempTargetForKL");
            ArffLoader arff3 = new ArffLoader();
            arff3.setFile(new File(generatedTargetData));
            Instances  targetData = arff3.getDataSet();

            double KL = getKL(targetData,sourceData);
            averageKL = averageKL + KL ;
            //System.out.println("KL:"+KL);
        } //end round
        averageKL = 1.0*averageKL/numberOfSimulation;
        System.out.println("TrueAverageKL:" + averageKL);

        return averageKL;
    }

    //Utility.getAverageKL(targetDataLoc+targetDataName,sourceDataSize,sourceModelLoc,sourceLearnedModelName,sourceSimulateDataLoc)
    public static double getAverageKL(String targetDataFile,String sourceSize, String sourceModelLoc,
                                      String sourceModelName, String generatedDataLoc, String sourceDataLoc, String sourceDataName)
            throws Exception{
        double averageKL = 0.0;
        ArffLoader arff = new ArffLoader();
        arff.setFile(new File(targetDataFile));
        Instances  targetData = arff.getDataSet();
        //Generate source data and calculate KL 10 times

        for (int round=1; round<=numberOfSimulation_Estimation; round++){
            String sampleSize = sourceSize;
            Random seedGenerator = new Random();
            String seed = seedGenerator.nextInt()+"";
            sourceModelName = sourceModelName.replace(".bif","");
            String modelName = new String(sourceModelName);
            //	String generatedSourceData = generatedDataLoc+modelName+"_seed"+seed+"_size"+sampleSize+".arff";
            //	generateData(sampleSize, seed, sourceModelLoc, modelName,
            //			generatedDataLoc, modelName+"_seed"+seed+"_size"+sampleSize);

            String generatedSourceData = generatedDataLoc+ "tempForKL.arff";
            generateData(sampleSize, seed, sourceModelLoc, modelName, generatedDataLoc, "tempForKL");
            ArffLoader arff2 = new ArffLoader();
            arff2.setFile(new File(generatedSourceData));
            Instances  sourceData = arff2.getDataSet();
            double KL = getKL(targetData,sourceData);
            averageKL = averageKL + KL ;
            //System.out.println("KL:"+KL);
        } //end round
        averageKL = 1.0*averageKL/numberOfSimulation_Estimation;
        System.out.println("averageKL:" + averageKL);

        //print true KL.
//		ArffLoader arff3 = new ArffLoader();
//		arff3.setFile(new File(sourceDataLoc+sourceDataName));
//		Instances  sourceData3 = arff3.getDataSet();
//		double trueKL = getKL(targetData,sourceData3);
//		System.out.println("trueKL:"+ trueKL);

        return averageKL;
    }

    public static double getKL(String sourceDataFile, String targetDataFile)
            throws Exception{
        ArffLoader arff1 = new ArffLoader();
        arff1.setFile(new File(targetDataFile));
        Instances  targetData = arff1.getDataSet();
        ArffLoader arff2 = new ArffLoader();
        arff2.setFile(new File(sourceDataFile));
        Instances  sourceData = arff2.getDataSet();
        double KL = getKL(targetData,sourceData);
        System.out.println("trueKL:" + KL);
        return KL;
    }


//	public static double getKL(Instances targetData, String modelfileLoc,
//			String sourceModelName, int sourceSize,String datafileLoc) throws Exception{
//		Random seedGenerator = new Random();
//		String seed = seedGenerator.nextInt()+"";
//		generateData(""+sourceSize, seed, "00experiment/priorModel/", sourceModelName,
//				"00experiment/priorModel/simulatedData/", sourceModelName+"_seed"+seed+"_size"+sourceSize);
//	//	ArffLoader arff2 = new ArffLoader();
//	//	arff2.setFile(new File(fileLocation+"BrainTumor_True_Source_Noisy1_seed-1434193311_size500.arff"));
//	//	arff2.setFile(new File(fileLocation+"BrainTumor_True_Target_seed-1767683974_size2000.arff"));
//	//	Instances  sourceData = arff2.getDataSet();
//		return 0.0;
//	}
//


    public static void generateData(String sampleSize, String seed, String modelLoc, String modelName, String dataLoc, String dataName) throws Exception{
        String[] options = new String[6];
        options[0] = new String("-M");
        options[1] = new String(sampleSize);   	//options[1] = new String("10");
        options[2] = new String("-S");
        options[3] = new String(seed);  //options[3] = new String("123");
        options[4] = new String("-F");
        options[5] = new String(modelLoc + modelName+".bif");   //options[5] = new String("model/BrainTumor_True_Target_Noisy1.bif");
        BayesNetGenerator b = new BayesNetGenerator();
        b.setOptions(options);
        b.generateRandomNetwork(); //Here, it is just initiate the model.
        b.generateInstances();
        PrintWriter printout = new PrintWriter(new File(dataLoc +dataName+".arff"));
        printout.println(b.toString());
        printout.close();
        //System.out.println(b.toString());
    }



    /**
     * Return KL (A|B) = SUM{P(A)*log[P(A)/P(B)]}
     * @param datasetA
     * @param datasetB
     * @return
     * @throws Exception
     */
    public static double getKL(Instances datasetA, Instances datasetB) throws Exception {
        ArrayList<String> intersectList = getInterSect(datasetA, datasetB);
        //System.out.println("intersectList:"+intersectList.size());
        HashMap<String,ArrayList<Integer>> countTable = getCountTable(intersectList,datasetA, datasetB);
        //	printTable(countTable);
        String[] keyList = countTable.keySet().toArray(new String[0]);
        double kl = 0.0;
        int totalCountA = datasetA.numInstances();
        int totalCountB = datasetB.numInstances();
        for (int i=0; i<keyList.length; i++){
            ArrayList<Integer> countList = countTable.get(keyList[i]);
            Integer pCount = countList.get(0);
            Integer qCount = countList.get(1);
            Integer totalCount = pCount + qCount;
            double p = 1.0 * pCount / totalCountA;
            double q = 1.0 * qCount / totalCountB;
            double value = 0.0;
            if (q == 0.0){
                q = 0.00000001;
            }
            if (p == 0.0) {
                value = 0.0;
            }
            else value = p * Math.log(p / q);
            //	System.out.println(p+","+q + ","+value);
            kl += value;
        }
        return kl;
    }



    /**
     * Return a variable, KL table
     * For node A, class C, calculate KL = sumc suma p(A=a,C=c)* log[p(A=a|C=c) / q(A=a|C=c)]
     * Assuming naive bayes structure and calculate KL
     * @param datasetA
     * @param datasetB
     * @return
     * @throws Exception
     */
    public static HashMap<String,Double> getEachKL(String classNodeName, Instances targetData,
                                                   Instances sourceData) throws Exception {
        HashMap<String,Double> klTable = new HashMap<String,Double>();

        HashMap<String,Integer> nameIndexTableSource = new HashMap<String,Integer>();
        for (int i=0; i<sourceData.numAttributes(); i++){
            String attName = sourceData.attribute(i).name();
            nameIndexTableSource.put(attName, i);
        }
        HashMap<String,Integer> nameIndexTableTarget = new HashMap<String,Integer>();
        for (int j=0; j<targetData.numAttributes(); j++){
            String attName = targetData.attribute(j).name();
            nameIndexTableTarget.put(attName, j);
        }

        ArrayList<String> classNodeValueList = new ArrayList<String>();
        Attribute targetSource = sourceData.attribute(nameIndexTableSource.get(classNodeName));
        for (int i=0; i<targetSource.numValues(); i++){
            String value = targetSource.value(i);
            if (!classNodeValueList.contains(value)) { classNodeValueList.add(value); }
        }
        Attribute targetTarget = targetData.attribute(nameIndexTableTarget.get(classNodeName));
        for (int j=0; j<targetTarget.numValues(); j++){
            String value = targetTarget.value(j);
            if (!classNodeValueList.contains(value)) { classNodeValueList.add(value); }
        }
        int totalCountTarget = targetData.numInstances();
        int totalCountSource = sourceData.numInstances();
        //calculate KL for the classNode
        double kl = 0.0;
        ArrayList<String> currentList = new ArrayList<String>();
        currentList.add(classNodeName);
        HashMap<String,ArrayList<Integer>> countClassTable = getCountTable(currentList,targetData, sourceData);
        printTable(countClassTable);
        for (int i=0; i<classNodeValueList.size();i++){
            String classNodeValueJ = classNodeValueList.get(i);
            Integer pCount = 0;
            Integer qCount = 0;
            if (countClassTable.containsKey(classNodeValueJ)){
                ArrayList<Integer> countList = countClassTable.get(classNodeValueJ);
                pCount = countList.get(0);
                qCount = countList.get(1);
            }
            double p = 1.0 * pCount / totalCountTarget;
            double q = 1.0 * qCount / totalCountSource;
            double value = 0.0;
            if (q == 0.0){
                q = 0.00000001;
            }
            if (p == 0.0) {
                value = 0.0;
            }
            else value = p * Math.log(p / q);
            kl += value;
            System.out.println(classNodeValueJ+","+pCount+","+qCount);
        }
        klTable.put(classNodeName, kl);

        //calculate KL for the other nodes (not class node)
        ArrayList<String> sharedNodeList = getInterSect(targetData,sourceData);
        sharedNodeList.remove(classNodeName);
        //System.out.println("sharedNodeList:"+sharedNodeList.size());
        for (int v=0; v<sharedNodeList.size(); v++){
            String oneVarName = sharedNodeList.get(v);
            currentList = new ArrayList<String>();
            currentList.add(oneVarName);
            currentList.add(classNodeName);
            HashMap<String,ArrayList<Integer>> countTable = getCountTable(currentList,targetData, sourceData);
            //	printTable(countTable);
            String[] keyList = countTable.keySet().toArray(new String[0]);
            kl = 0.0;
            for (int i=0; i<keyList.length; i++){
                String oneConfig = keyList[i];
                String classNodeValueJ = oneConfig.split("\\|")[1];
                Integer pCountParent = countClassTable.get(classNodeValueJ).get(0);
                Integer qCountParent = countClassTable.get(classNodeValueJ).get(1);

                ArrayList<Integer> countList = countTable.get(oneConfig);
                Integer pCountJoint = countList.get(0);
                Integer qCountJoint = countList.get(1);

                double pJoint = 1.0 * pCountJoint / totalCountTarget;
                double pCond = 1.0 * pCountJoint / pCountParent;
                double qCond = 1.0 * qCountJoint / qCountParent;

                double value = 0.0;
                if (qCond == 0.0 | qCountParent==0.0){
                    qCond = 0.00000001;
                }
                if (pCountParent==0.0){
                    pCond = 0.0;
                }
                if (pJoint == 0.0) {
                    value = 0.0;
                }
                else value = pJoint * Math.log(pCond / qCond);
                System.out.println(oneVarName+"-"+oneConfig +",target:"+pCountJoint+","+pCountParent);
                System.out.println(oneVarName+"-"+oneConfig +",source:"+qCountJoint+","+qCountParent);
                System.out.println(pJoint+","+pCond+","+qCond + ","+value);
                kl += value;
            }
            klTable.put(oneVarName,kl);
        }
        return klTable;
    }


    public static ArrayList<String> getInterSect(Instances datasetA, Instances datasetB){
        ArrayList<String> nameListA = new ArrayList<String>();
        for (int i=0; i<datasetA.numAttributes(); i++){
            nameListA.add(datasetA.attribute(i).name());
        }
        ArrayList<String> nameListB = new ArrayList<String>();
        for (int i=0; i<datasetB.numAttributes(); i++){
            nameListB.add(datasetB.attribute(i).name());
        }
        for (int j=nameListA.size()-1; j>=0; j--){
            String oneVar = nameListA.get(j);
            if (!nameListB.contains(oneVar)){
                nameListA.remove(j);
            }
        }
//		for (int i=0; i<nameListA.size();i++){
//			System.out.print(nameListA.get(i));
//		}
//		System.out.println();
        return nameListA;
    }



    public static HashMap<String,ArrayList<Integer>> getCountTable(ArrayList<String> intersectList,
                                                                   Instances datasetA, Instances datasetB) throws Exception{
        HashMap<String,ArrayList<Integer>> countTable = new HashMap<String,ArrayList<Integer>>();
        HashMap<String,Integer> nameIndexTableA = new HashMap<String,Integer>();
        for (int k=0; k<datasetA.numAttributes(); k++){
            String varName = datasetA.attribute(k).name();
            nameIndexTableA.put(varName, k);
        }
        HashMap<String,Integer> nameIndexTableB = new HashMap<String,Integer>();
        for (int k=0; k<datasetB.numAttributes(); k++){
            String varName = datasetB.attribute(k).name();
            nameIndexTableB.put(varName, k);
        }

        for (int i=0; i<datasetA.numInstances();i++){
            Instance oneInstance = datasetA.instance(i);
            String oneInstanceString = getConfig(intersectList, oneInstance, nameIndexTableA);
            if (countTable.containsKey(oneInstanceString)){
                ArrayList<Integer> countList = countTable.get(oneInstanceString);
                countList.set(0,countList.get(0)+1);
            }
            else //firstly appear config
            {
                ArrayList<Integer> countList = new ArrayList<Integer>();
                countList.add(1); countList.add(0);
                countTable.put(oneInstanceString, countList);
            }
        }
        for (int i=0; i<datasetB.numInstances();i++){
            Instance oneInstance = datasetB.instance(i);
            String oneInstanceString = getConfig(intersectList, oneInstance, nameIndexTableB);
            if (countTable.containsKey(oneInstanceString)){
                ArrayList<Integer> countList = countTable.get(oneInstanceString);
                countList.set(1,countList.get(1)+1);
            }
            else //firstly appear config
            {
                ArrayList<Integer> countList = new ArrayList<Integer>();
                countList.add(0); countList.add(1);
                countTable.put(oneInstanceString, countList);
            }
        }
        return countTable;
    }

    private static String getConfig(ArrayList<String> intersectList, Instance oneInstance, HashMap<String,Integer> nameIndexTable) throws Exception{
        String oneInstanceString = new String("");
        for (int i=0; i<intersectList.size(); i++){
            String varName = intersectList.get(i);
            if (!nameIndexTable.containsKey(varName)) {throw new Exception("not contain the variable!"); }
            int indexOfVar = nameIndexTable.get(varName);
            String attValue = oneInstance.toString(indexOfVar);
            oneInstanceString = oneInstanceString + attValue;
            if (i<intersectList.size()-1){
                oneInstanceString = oneInstanceString + "|" ;
            }
        }
        return oneInstanceString;
    }



    public static void printTable(HashMap<String,ArrayList<Integer>> countTable){
        String[] keyList = countTable.keySet().toArray(new String[0]);
        for (int i=0; i<keyList.length; i++){
            String key = keyList[i];
            ArrayList<Integer> countList = countTable.get(key);
            System.out.println(key+"," + countList.get(0) + "," + countList.get(1));
        }
    }









    public static String getConfig(String item, String configFileName) throws FileNotFoundException{
        Scanner input = new Scanner(new File(configFileName));
        String output = new String("");
        while (input.hasNext()){
            String oneLine = input.nextLine();
//			System.out.println(oneLine);
            if (oneLine.length()>0){
                String[] values = oneLine.split("=");
                if (values[0].equals(item)){
                    if (values.length==2)
                        return values[1];
                    else return output;
                }
            }
        }
        input.close();
        return output;
    }

}

