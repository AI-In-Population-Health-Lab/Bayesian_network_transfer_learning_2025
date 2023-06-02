import java.io.File;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import Utility.Utility;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;

public class FeatureSelection {
    static String configFileName;
    static String sourceModelLoc,sourceLearnedModelName,sourceTrueModelName,sourceCleanInjectModelXML,
            sourceClearnInjectModelXDSL,sourceSimulateDataLoc,sourceSimulateDataName,sourceDataLoc,sourceDataName,sourceDataSize,
            targetModelLoc,targetTrueModelName,targetLearnedModelName,targetFinalModelName,targetDataLoc,targetDataName,targetNodeName,
            targetTestDataName,resultLoc,resultProbName,resultAUCName,resultCalibrationName,
            utilityLoc,temporaryFileName,hashCodeName,logLoc,logName,targetClassName;

    public static void main(String[] args) throws Exception {
        test();
//		testCode();
//		testAverageInfoCode();
    }

    public static void test() throws Exception{
        configFileName = new String("00experiment/utility/Configuration.txt");
        setConfig();
        ArffLoader arff = new ArffLoader();
        arff.setFile(new File(sourceDataLoc+sourceDataName));
        Instances instances = arff.getDataSet();
        int targetIndex = instances.numAttributes()-1;
        instances.setClassIndex(targetIndex);
        //	Instances cfsFiltedData  = useFilterCfsSubsetEval(instances) ;
        Instances infogainFiltedData  = useFilterInfoGain(instances, new Double("0.001"));
    }

    public static void testCode() throws Exception{
        configFileName = new String("00experiment/utility/Configuration.txt");
        setConfig();
        ArffLoader arff = new ArffLoader();
        arff.setFile(new File(targetDataLoc+targetDataName));
        Instances instances = arff.getDataSet();
        int targetIndex = instances.numAttributes()-1;
        for (int i=0; i<instances.numAttributes(); i++){
            String name = instances.attribute(i).name();
            if (name.equals(targetClassName)){
                targetIndex = i;
                break;
            }
        }
        HashMap<String,Double> table = getInfogain(instances, targetIndex) ;
    }

    public static Instances useFilterCfsSubsetEval(Instances trainData) throws Exception {
        int targetIndex = trainData.numAttributes()-1;
        for (int i=0; i<trainData.numAttributes(); i++){
            String name = trainData.attribute(i).name();
            if (name.equals(targetClassName)){
                targetIndex = i;
                break;
            }
        }
        trainData.setClassIndex(targetIndex);
        weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
		    /*BestFirst search = new BestFirst();
		    String optionString = "-D 0 -N 5";
			String[] options = optionString.split(" ");
			search.setOptions(options);*/
        GreedyStepwise search = new GreedyStepwise();
        /*   search.setSearchBackwards(true);*/
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(trainData);
        Instances filtedTrainData = Filter.useFilter(trainData, filter);
        System.out.println(filtedTrainData.toSummaryString());;
        for (int k=0; k<filtedTrainData.numAttributes(); k++){
            System.out.println(filtedTrainData.attribute(k).name());
        }
        return filtedTrainData;
        //System.out.println(newData);
    }


    public static Instances useFilterInfoGain(Instances trainData, double threshold) throws Exception {
        int targetIndex = trainData.numAttributes()-1;
        for (int i=0; i<trainData.numAttributes(); i++){
            String name = trainData.attribute(i).name();
            if (name.equals(targetClassName)){
                targetIndex = i;
                break;
            }
        }
        trainData.setClassIndex(targetIndex);
        weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        //must use ranker method
        Ranker search = new Ranker();
        search.setThreshold(threshold);
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(trainData);
        Instances filtedTrainData = Filter.useFilter(trainData, filter);
        System.out.println(filtedTrainData.toSummaryString());
        for (int i=0; i<trainData.numAttributes(); i++)
        {
            //System.out.println(trainData.attribute(i).name()  + "," + eval.evaluateAttribute(i));
        }
        for (int k=0; k<filtedTrainData.numAttributes(); k++){
            //System.out.println(filtedTrainData.attribute(k).name());
        }
        return filtedTrainData;
        //System.out.println(newData);
    }


    public static void testAverageInfoCode() throws Exception{
        configFileName = new String("00experiment/utility/Configuration.txt");
        setConfig();
        ArffLoader arff = new ArffLoader();
        arff.setFile(new File(targetDataLoc+targetDataName));
        Instances instances = arff.getDataSet();
        int targetIndex = instances.numAttributes()-1;
        for (int i=0; i<instances.numAttributes(); i++){
            String name = instances.attribute(i).name();
            if (name.equals(targetClassName)){
                targetIndex = i;
                break;
            }
        }
        instances.setClassIndex(instances.numAttributes() - 1);
        Instances oneSample = instances;
        InfoGainAttributeEval infoEval = new InfoGainAttributeEval();
        infoEval.buildEvaluator(oneSample);
        for (int featureIndex=0; featureIndex<oneSample.numAttributes(); featureIndex++){
            if (featureIndex!=targetIndex){
                double infogainForFeature = infoEval.evaluateAttribute (featureIndex);
                System.out.println(oneSample.attribute(featureIndex).name()+":"+infogainForFeature);
            }
        }
        System.out.println("average:");
        int numberSamples = 1000;
        HashMap<String,Double> table = getAverageInfogain(instances, targetIndex,numberSamples) ;
        String[] keyList = table.keySet().toArray(new String[0]);
        for (int k=0; k<keyList.length; k++){
            String key = keyList[k];
            System.out.println(key+":"+table.get(key));
        }
    }



    public static HashMap<String,Double> getInfogain(Instances data, int targetIndex) throws Exception{
        HashMap<String,Double> infogainNameResultTable =  new HashMap<String,Double>();
        data.setClassIndex(data.numAttributes() - 1);
        InfoGainAttributeEval infoEval = new InfoGainAttributeEval();
        infoEval.buildEvaluator(data);
        for (int featureIndex=0; featureIndex<data.numAttributes(); featureIndex++){
            if (featureIndex!=targetIndex){
                String featureName = data.attribute(featureIndex).name();
                double infogainForFeature = infoEval.evaluateAttribute (featureIndex);
                infogainNameResultTable.put(featureName, infogainForFeature);
                System.out.println(featureName+":"+infogainForFeature);
            }
        }
        return infogainNameResultTable;
    }


    public static HashMap<String,Double> getAverageInfogain(Instances data, int targetIndex, int numberSamples) throws Exception{
        ArrayList<Instances> sampleList = getBootstrappingSamples(data,numberSamples);
        HashMap<Integer,Double> infogainTable =  new HashMap<Integer,Double>();
        for (int featureIndex=0; featureIndex<data.numAttributes(); featureIndex++){
            String featureName = data.attribute(featureIndex).name();
            if (featureIndex!=targetIndex){
                infogainTable.put(featureIndex, 0.0);
            }
        }
        for (int j=0; j<sampleList.size(); j++){
            Instances oneSample = sampleList.get(j);
            oneSample.setClassIndex(targetIndex);
            InfoGainAttributeEval infoEval = new InfoGainAttributeEval();
            infoEval.buildEvaluator(oneSample);
            for (int featureIndex=0; featureIndex<data.numAttributes(); featureIndex++){
                if (featureIndex!=targetIndex){
                    //	System.out.print(oneSample.attribute(featureIndex).name()+":");
                    double infogainForFeature = infoEval.evaluateAttribute (featureIndex);
                    //System.out.print(infogainForFeature+",");
                    double previousTotalInfoForFeature = infogainTable.get(featureIndex);
                    infogainTable.put(featureIndex,previousTotalInfoForFeature+infogainForFeature);
                    //System.out.println();
                }
            }
        }
        for (int featureIndex=0; featureIndex<data.numAttributes(); featureIndex++){
            if (featureIndex!=targetIndex){
                double totalInfoForFeature = infogainTable.get(featureIndex);
                double averageInfoForFeature = totalInfoForFeature/numberSamples;
                infogainTable.put(featureIndex,averageInfoForFeature);
            }

        }
        HashMap<String,Double> infogainNameResultTable =  new HashMap<String,Double>();
        for (int featureIndex=0; featureIndex<data.numAttributes(); featureIndex++){
            String featureName = data.attribute(featureIndex).name();
            if (featureIndex!=targetIndex){
                infogainNameResultTable.put(featureName, infogainTable.get(featureIndex));
            }
        }
        return infogainNameResultTable;
    }

    public static ArrayList<Instances> getBootstrappingSamples(Instances originalData, int numberSamples) throws Exception{
        ArrayList<Instances> sampleList = new ArrayList<Instances>();
        for (int i=0; i<numberSamples; i++){
            weka.filters.unsupervised.instance.Resample filter = new weka.filters.unsupervised.instance.Resample();
            Random seedGenerator = new Random();
            int newSeed =  seedGenerator.nextInt();
            filter.setRandomSeed(newSeed) ;
            filter.setInputFormat(originalData);
            //filter.setNoReplacement(new Boolean(true));
            Instances sample = Filter.useFilter(originalData,filter);
            sampleList.add(sample);
            //System.out.println(sample.size());
        }
        return sampleList;
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

        //	sourceLearnedModelName=Utility.getConfig("sourceLearnedModelName",configFileName);
        //	targetLearnedModelName=Utility.getConfig("targetLearnedModelName",configFileName);
        //	targetFinalModelName=Utility.getConfig("targetFinalModelName",configFileName);

        sourceLearnedModelName=sourceDataName.replace(".arff", ".bif");
        targetLearnedModelName=targetDataName.replace(".arff", ".bif");
        targetFinalModelName=sourceLearnedModelName.replace(".bif","")+"-"+targetDataName.replace(".arff", ".bif");


    }


}
