/*
 *   This program is under development
 */

/*
 * EBMC.java
 * Copyright (C) 2015 University of Pittsburgh, PA, USA
 *
 */
package Utility;
import java.io.*;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import smile.Network;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.bayes.net.EditableBayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.classifiers.bayes.net.estimate.BMAEstimator;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.DiscreteEstimatorBayes;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.BayesNet;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.converters.ArffLoader;

import org.joda.time.DateTime;

import smile.Network;
import smile.SMILEException;
import edu.pitt.dbmi.dataset.ConvertBIFtoGENIE;
import edu.pitt.isp.sverchkov.smile.SMILEBayesNet;
import weka.estimators.Estimator;



/*****
 * This is the most important code. BTLSM9_new_KL.
 * ******/

/**
 * <!-- globalinfo-start --> This algorithm performs greedy search in a subspace
 * of Bayesian Networks to find the one that best predicts a target node.<br/>
 * <br/>
 * For more information refer to:<br/>
 * <br/>
 * G. F. Cooper, P. Hennings-Yeomans, S. Visweswaran, & M. Barmada, (2010). An
 * efficient Bayesian method for predicting clinical outcomes from genome-wide
 * data. AMIA Annual Symposium Proceedings. 127-131.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;inproceedings{Cooper 2010,
 *    author = {G.F. Cooper, P. Hennings-Yeomans, S. Visweswaran, & M. Barmada},
 *    pages = {127-131},
 *    publisher = {AMIA Annual Symposium Proceedings},
 *    title = {An efficient Bayesian method for predicting clinical outcomes from genome-wide data},
 *    year = {2010}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author  Ye Ye (yey5@pitt.edu)
 * with functions from Arturo Lopez Pineda (arl68@pitt.edu) and Kevin V. Bui (kvb2@pitt.edu)
 * @version $Revision: 1.0 $
 */
public class BNTL extends SearchAlgorithm implements TechnicalInformationHandler {
    /** points to Bayes network for which a structure is searched for **/
    static BayesNet m_BayesNet; //Current target Bayesian network. This one is initiated with source_clean_BayesNet, and will be updated after searching finish.
    static BayesNet source_BayesNet; //source Bayesian network
    static BayesNet source_clean_BayesNet; //source Bayesian network after removing unexisting features(for simulation data to get N')
    static BayesNet source_clean_inject_BayesNet; //This Bayesian network is for Nijk' calculation purpose.
    static HashMap<String,Integer> nameIndexTable;
    static Instances targetInstances;
    static Instances sourceInstances;
    static SMILEBayesNet  smileNet; //a smile network source clean network
    static int[][] candiateParents;
    static Instances simulatedSourceInstance;
    static double trueKL, avgKL;
    static HashMap<String,Double> trueBFWeightTable, avgBFWeightTable;

    static double adjustedPriorSampleSize; // note: see page 18 in thesis, N' can be assessed as the number of observations would have been seen in order to have the same confidence as our prior knowledge.
    /*** The prior equivalent sample size value, when using BDEu */
    private int m_SourceSampleSize;
    static HashMap<String,Double> weightTable;

    static ArrayList<String> sharedNodeList;
    static String configFileName;
    static String sourceModelLoc,sourceLearnedModelName,sourceLearnedCleanedModelName, sourceTrueModelName,sourceCleanInjectModelXML,
            sourceClearnInjectModelXDSL,sourceSimulateDataLoc,sourceSimulateDataName,sourceDataLoc,sourceDataName,sourceDataSize,
            targetModelLoc,targetTrueModelName,targetLearnedModelName,targetFinalModelName, targetDataLoc,targetDataName,targetNodeName,
            targetTestDataName,resultLoc,resultProbName,resultAUCName,resultCalibrationName,
            utilityLoc,temporaryFileName,hashCodeName,logLoc,logName, targetClassName;
    static String featureSelectionApproach, modelLearningApproach;
    static String transferLearningApproach, transferLearningWeight;
    static Boolean useTrueSourceDataForWeight;

    static double KL_targetData_learnedSourceModel, KL_targetData_trueSourceModel;
    static HashMap<String,Double> learnedSource_nodeKLTable;
    static HashMap<String,Double> trueSource_nodeKLTable;

    protected static boolean print = false;
    protected static final int MAXNEWCHILDREN = 10;  // maximum new children that any node is allowed to have in a "rule"
    protected static final int MININUM_EXPONENT = -1022;
    /**
     * Holds prior on count
     */
    //double m_fAlpha = 0.5;
    protected static final double MINUS_INFINITY =  -1.0e308;

    /** for serialization */
    static final long serialVersionUID = 6176545934752116631L;

    protected byte[][] cases;  // contains all the training (and possibly testing) data, the first index is for instance, the second index is for feature
    protected boolean childJustAddedFlag = false;
    protected boolean[] childPresent; //  note: it is an array, recording whether a node is a child of the target node
    protected boolean[] parentPresent; //  note: it is an array, recording whether a node is a parent of the target node

    protected int[] children; //Ye: children list of the target node
    protected int[] counts;  // the counts at the leaves of the tree created by FileCase

    protected int[] countsTree;  // the counts at the leaves of the tree created by FileCase
    protected int countsTreePtr, countsPtr;
    protected FileCaseRecord[] fileCaseCache;

    protected int firstCase, lastCase;
    protected Instances inst;

    protected double[][][] lnChildProb;
    protected double[][] lnChildrenProb;

    protected double lnTotalProb;
    protected int lowerBound;  // usually 1 -- first case

    protected String my_model= "";


    /*** Override the initialization as naive Bayes to False */
//	private final boolean m_bInitAsNaiveBayes = false;
    /*** Overrides the Markov blanket correction */
    //private final boolean m_bMarkovBlanketClassifier = false;
    /*** Overrides the scoring metric to prequential */
    //private int m_nScoreType = 5;
    /*** The expected parents of target  */
    private int m_ExpectedParentsOfTarget=2;
    /*** The maximum number of children that any node is allowed to have */
    private int m_MaxNrOfChildren=2;

    /*** The maximum number of parents that any node is allowed to have */
    private int m_MaxNrOfParents=2;

    /*** The scoring metric to use */
    private int m_ScoreMetric = 0; // 0 is K2, 1 is BDeu, 2 is prior network method

    /*** The adjust sample size approach */
    private int m_AdjustSourceSampleSizeMetric = 0; // 0 is unchange,
    //1 is sample size ratio; 2 is KL; 3 is Bayes factor; 4 is KL_targetData_learnedSourceModel;
    //5 is KL_targetData_trueSourceModel; 6 is nodeKLTable_targetData_learnedSourceModel; 7 is nodeKLTable_targetData_trueSourceModel
    //20 is adjust_KL_targetData_trueSourceModel; 21 adjust_KL_targetData_learnedSourceModel

    protected int[] map; //map[child] = numberOfChildren; the order of the child of the target node. for example X4 is the 2nd child of the target map[4]=2

    protected int maxCell;
    protected int maxChildren;
    protected int maxParents;

    protected int maxValue;
    protected boolean[] newChildPresent; // note:  in addition to the fixed children

    protected int[] newChildren; // note: in addition to the fixed children

    protected byte[] nodeDimension;

    protected NodeInfoRecord[] nodeInfo;


    protected double[] nodeProb; //  note: nodeProb[nodeValue] = P(node = nodeValue | parents_node, case1,...,casei-1)

    protected double nodeScore; //  note: Score the targetnode with current children (if any), print out the score in output
    protected int numberOfCases; //  note: number of instances
    protected int numberOfChildren; //  note: current number of children of the target node
    protected int numberOfModelsScored;

    protected int numberOfNodes; //  note: number of features. Each feature is a node, although the node may not be in the Markov Blanket of the target node.
    protected int[][] parents;  // note: parents[i, 0] represents the number of parents of node i

    private Tag[] SCORING_METRICS={
            new Tag(0, "K2"),
            new Tag(1, "BDeu"),
            new Tag(2, "Prior Model")
    };


    protected int[] targetCounts; //note: only initiate, did not use it.



    public int targetNode;  // the outcome node being predicted
    protected int totalModelsScored;



    // the range of cases for training and testing
    protected int upperBound;  // usually the number of cases

    protected int[] values;  // for a given case, it contains the values of the parents of the target and then value of the target

    protected DateTime startTime;

    /*new variables Ye added*/

    protected int numberOfNodesSource; //  note: number of features. Each feature is a node, although the node may not be in the Markov Blanket of the target node.


    /**
     * default constructor
     */
    public BNTL() {
    } // c'tor

    /**
     * default constructor
     */
    public BNTL(int predictors, int max_parents, int max_children) {
        setExpectedParentsOfTarget(predictors);
        setMaxNrOfParents(max_parents);
        setMaxNrOfChildren(max_children);
    } // c'tor

    /**
     * default constructor
     */
    public BNTL(int predictors, int max_parents, int max_children, int pess) {
        setExpectedParentsOfTarget(predictors);
        setMaxNrOfParents(max_parents);
        setMaxNrOfChildren(max_children);
        //setScoreMetric(new SelectedTag("BDeu", SCORING_METRICS));
        m_ScoreMetric=1; //for BDeu
        //setPriorEquivalentSampleSize(pess);
    } // c'tor




    /**
     * Initiate with source model
     * @param predictors
     * @param max_parents
     * @param max_children
     * @param sourceModelFile
     * @throws Exception
     */
    //Ye added
    public BNTL(String configFile, String scoreMethod, String adjustMethod,
                String startNetwork,  double trueKL1, HashMap<String,Double> trueBFWeightTable1,
                double avgKL1, HashMap<String,Double> avgBFWeightTable1) throws Exception {

        trueKL = trueKL1;
        trueBFWeightTable = trueBFWeightTable1;
        avgKL = avgKL1;
        avgBFWeightTable = avgBFWeightTable1;

        configFileName = configFile;
        System.out.println("Configuration File is " + configFileName);
        System.out.println("Start network is " + startNetwork);
        transferLearningApproach = scoreMethod;
        transferLearningWeight = adjustMethod;
        setConfig();
        useTrueSourceDataForWeight = new Boolean("False");
        targetFinalModelName=startNetwork.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";

        sourceLearnedCleanedModelName = startNetwork.replace(".bif", "")+"_cleaned.bif";

        System.out.println("It was developed with " + sourceDataSize + " instances");
        if (scoreMethod.equals("priorModelApproach")){
            m_ScoreMetric=2;
        }
        else if (scoreMethod.equals("K2")){
            m_ScoreMetric=0;
        }
        else if (scoreMethod.equals("BDeu")){
            m_ScoreMetric=1;
        }
        if (adjustMethod.equals("unadjust")){
            m_AdjustSourceSampleSizeMetric=0;
        }
        else if (adjustMethod.equals("ratio")){
            m_AdjustSourceSampleSizeMetric=1;
        }
        else if (adjustMethod.equals("KL")){
            m_AdjustSourceSampleSizeMetric=2;
        }
        else if (adjustMethod.equals("BayesFactor")){
            m_AdjustSourceSampleSizeMetric=3;
        }
        else if (adjustMethod.equals("trueKL")){
            m_AdjustSourceSampleSizeMetric=2;
            useTrueSourceDataForWeight = new Boolean("True");
        }
        else if (adjustMethod.equals("trueBayesFactor")){
            m_AdjustSourceSampleSizeMetric=3;
            useTrueSourceDataForWeight = new Boolean("True");
        }
        else if (adjustMethod.equals("KL_targetData_learnedSourceModel")){
            m_AdjustSourceSampleSizeMetric=4;
        }
        else if (adjustMethod.equals("KL_targetData_trueSourceModel")){
            m_AdjustSourceSampleSizeMetric=5;
            useTrueSourceDataForWeight = new Boolean("True");
        }
        else if (adjustMethod.equals("nodeKLTable_targetData_learnedSourceModel")){
            m_AdjustSourceSampleSizeMetric=6;
        }
        else if (adjustMethod.equals("nodeKLTable_targetData_trueSourceModel")){
            m_AdjustSourceSampleSizeMetric=7;
            useTrueSourceDataForWeight = new Boolean("True");
        }

        else if (adjustMethod.equals("adjust_KL_targetData_trueSourceModel")){
            m_AdjustSourceSampleSizeMetric=20;
            useTrueSourceDataForWeight = new Boolean("True");
        }
        else if (adjustMethod.equals("adjust_KL_targetData_learnedSourceModel")){
            m_AdjustSourceSampleSizeMetric=21;
        }


        source_BayesNet = new BIFReader();
        try {
            ((BIFReader) source_BayesNet).processFile(sourceModelLoc+startNetwork);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        m_SourceSampleSize = Integer.parseInt(sourceDataSize);

        //Later, need to simulate source data, write methods to adjust sample size
        System.out.println("\n\n==========================ORIGINAL SOURCE MODEL==========================");
        System.out.println(print(source_BayesNet));
        System.out.println(source_BayesNet);

    } // c'tor


    /**
     * search determines the network structure/graph of the network
     * based on the EBMC search algorithm
     *
     * @param bayesNet the network, This bayesNet is an empty net, coupled with the target Training data
     * @param targetTrainingData the data to work with
     * @throws Exception if something goes wrong
     */
    public void buildStructure(BayesNet bayesNet, Instances targetTrainingData) throws Exception {
        targetInstances = targetTrainingData;

        System.out.println("\n==========================CLEANING SOURCE MODEL==========================");
        m_BayesNet = bayesNet; //Now, m_BayesNet and bayesNet is empty and m_instances are
        //	System.out.println("m_BayesNet:");

        /*Part A. clean source model:
         * 1.remove unobserved node: cleanSourceBN
         * 2. move parentsToChildren: : has not done yet.
         * 3. inject target specific feature.
         * need to modify the injectTargetSpecificFeature method (add link from class to each injection)
         * need to create a new method to save BayesNet in genie format.
         * 				   need to create prior network from source_clean_inject_BayesNet in genie format
         * 5. read prior network for CPT part
         * */
        //	System.out.println("Source:");
        //	System.out.println(source_BayesNet.graph());
        //remove nodes that do not appear in target data
        source_clean_BayesNet = cleanSourceBN(source_BayesNet, targetTrainingData);
//		System.out.println("Source_clean:");
//		System.out.println(source_clean_BayesNet.graph());
        PrintWriter printout = new PrintWriter(new File(sourceModelLoc+sourceLearnedCleanedModelName));
        printout.println(source_clean_BayesNet.graph());
        printout.flush();
        printout.close();

        getSharedList(source_clean_BayesNet);

//		System.out.println(source_clean_BayesNet);

        //inject target specific features and add dummy links between these features and the class
        source_clean_inject_BayesNet = injectTargetSpecificFeature(source_clean_BayesNet, targetTrainingData);
//		System.out.println("Source_clean_inject:");
//		System.out.println("source_clean_inject_BayesNet:");
//		System.out.println(source_clean_inject_BayesNet);
//		System.out.println(source_clean_inject_BayesNet.graph());

        Network net = new Network();
        // 	net.readFile("00experiment/priorModel/BrainTumor_Prior_Network.xdsl");
//	 	net.readFile("00experiment/intermediate/source_clean_inject.xdsl");
        net.readFile(sourceModelLoc+sourceClearnInjectModelXDSL);
        boolean convertIDs = false;
        smileNet = new SMILEBayesNet(net, convertIDs);


        /*Part B. Getting target model
         * 1. initiate targetInstances and targetNode, max_parent, max_children.
         * The latter two variables are just used to restrict search when the number of candidate variables is extremely large.
         * 2. initiate numberOfNodes, nodeDimension array, numberOfCases, initiate the size of the cases[][]
         * */


        adjustSourceSampleSize();
        targetNode = targetInstances.numAttributes();
        //targetNode = targetInstances.classIndex();
        int max_parents = targetInstances.numAttributes()-1;
        int max_children = targetInstances.numAttributes()-1;
        setMaxNrOfParents(max_parents);
        setMaxNrOfChildren(max_children);

//***************insert sample size adjustment method here.
        // when generating data, make sure class node is the last Node

        /*Part 2. grow from source model*/
        //initiate everything
        readCases();
        initVariables();

        updateParents(source_clean_BayesNet);
        System.out.println("\n\n==========================CLEANED SOURCE MODEL==========================");
        System.out.println(String.format("score: %f", scoreNode(targetNode)));
        //This net keeps all parents set from source_clean_BayesNet and also include all target features. It is similar to the source_clean_inject.
        updateBayesNet(targetTrainingData);
//		m_BayesNet.estimateCPTs(); //this is using simple estimator
//		System.out.println(print(m_BayesNet));
//		System.out.println(m_BayesNet);

        if (m_ScoreMetric==0){
            System.out.println("ScoreMetric: K2"); //0 is K2 method
        }
        else if (m_ScoreMetric==1){
            System.out.println("ScoreMetric: BDeu"); //1 is K2 method
        }
        else if (m_ScoreMetric==2){
            System.out.println("ScoreMetric: Prior Model");	// 2 is the prior model approach
        }

        if (m_AdjustSourceSampleSizeMetric==0){
            System.out.println("Adjust method: unadjust");
            System.out.println("source sample:"+ adjustedPriorSampleSize);
        }
        else if (m_AdjustSourceSampleSizeMetric==1){
            System.out.println("Adjust method: ratio");
            System.out.println("adjusted source sample:"+ adjustedPriorSampleSize);
        }
        else if (m_AdjustSourceSampleSizeMetric==2){
            System.out.println("Adjust method: KL");
            System.out.println("adjusted source sample:"+ adjustedPriorSampleSize);
        }
        else if (m_AdjustSourceSampleSizeMetric==4){
            System.out.println("Adjust method: new KL using target data and learned source model");
            System.out.println("adjusted source sample:"+ adjustedPriorSampleSize);
        }
        else if (m_AdjustSourceSampleSizeMetric==5){
            System.out.println("Adjust method: new KL using target data and true source model");
            System.out.println("adjusted source sample:"+ adjustedPriorSampleSize);
        }
        else if (m_AdjustSourceSampleSizeMetric==20){
            System.out.println("Adjust method: adjust new KL using target data and true source model,"
                    + " divided by target data size");
            System.out.println("adjusted source sample:"+ adjustedPriorSampleSize);
        }
        else if (m_AdjustSourceSampleSizeMetric==21){
            System.out.println("Adjust method: adjust new KL using target data and learned source model"
                    + " divided by target data size");
            System.out.println("adjusted source sample:"+ adjustedPriorSampleSize);
        }

        else if (m_AdjustSourceSampleSizeMetric==3){
            System.out.println("Adjust method: BayesFactor");
            System.out.println("adjusted source sample:");
            for (int i=0; i<sharedNodeList.size(); i++){
                String key = sharedNodeList.get(i);
                System.out.println(key+","+weightTable.get(key)*m_SourceSampleSize);
            }
        }
        else if (m_AdjustSourceSampleSizeMetric==6){
            System.out.println("Adjust method: new KL for each node, using target data and learned source model");
            System.out.println("adjusted source sample:");
            for (int i=0; i<sharedNodeList.size(); i++){
                String key = sharedNodeList.get(i);
                System.out.println(key+","+weightTable.get(key)*m_SourceSampleSize);
            }
        }
        else if (m_AdjustSourceSampleSizeMetric==7){
            System.out.println("Adjust method: new KL for each node, using target data and true source model");
            System.out.println("adjusted source sample:");
            for (int i=0; i<sharedNodeList.size(); i++){
                String key = sharedNodeList.get(i);
                System.out.println(key+","+weightTable.get(key)*m_SourceSampleSize);
            }
        }

        System.out.println("\n==========================DEVELOPING TARGET MODEL==========================");
        boolean hopeRemains = true;
        startTime = new DateTime(System.currentTimeMillis());

        System.out.println("\n------Pruning for initial model");
        weedoutWeakArcs();	//firstly, prune the inital model
//		updateBayesNet(targetTrainingData);
//		m_BayesNet.estimateCPTs();
//		System.out.println(print(m_BayesNet));
//		System.out.println(m_BayesNet); //
        //	updateBayesNet(); // update parent set of the bayesian network based on parents[][]
        System.out.println(String.format("After pruning == score: %f", scoreNode(targetNode)));

        /*Part 3. start searching from model*/
        //EBMC1 Algorithm
        int numberGraph=0;

        while (hopeRemains && (numberOfChildren < maxChildren)) {
            numberGraph++;
            System.out.println("\n------Searching parents");
            searchParentsEBMC1(targetNode);
            System.out.println(String.format("After searching Parent == score: %f", scoreNode(targetNode)));
            System.out.println(printParents());
            if ((numberOfChildren + parents[targetNode][0]) > maxChildren) {
                hopeRemains = false;
            } else {
                System.out.println("\n------Moving current parents to children");
                hopeRemains = moveParentsToChildren();
                System.out.println(String.format("After moveParentsToChildren == score: %f", scoreNode(targetNode)));
            }
            System.out.println(printParents());
            System.out.println("\n------Pruning after searching parents and moving parents to children");
            System.out.println(printParents());
            weedoutWeakArcs();
            System.out.println(String.format("After pruning == score: %f", scoreNode(targetNode)));
            System.out.println(printParents());
        }	   	//end while loop

        System.out.println("\n\n==========================FINAL MODEL==========================");
        System.out.println("score: " + scoreNode(targetNode));
        System.out.println("");
        updateBayesNet(targetTrainingData);

        /*Part 3. get CPTs*/
        getCPTCombineModelData();
        System.out.println(print(m_BayesNet));
        System.out.println(m_BayesNet);

//		PrintWriter printout = new PrintWriter(new File("00experiment/learnedModel/final.xml"));
        PrintWriter printout2 = new PrintWriter(new File(targetModelLoc+targetFinalModelName));
        printout2.println(m_BayesNet.graph());
        printout2.flush();
        printout2.close();


		/*Part 4. get Performance

		String testDataset = new String(targetDataLoc+targetTestDataName);
		Instances testData = new Instances(new BufferedReader(new FileReader(testDataset)));


		System.out.println("AUC_final_model:" + evaluateBN(m_BayesNet, testData));
	//	System.out.print(m_BayesNet.graph());

		System.out.println("AUC_source_clean:" + evaluateBN(source_clean_BayesNet, testData));
	//	System.out.print(source_clean_BayesNet.graph());

		System.out.println("AUC_source_clean_inject:" + evaluateBN(source_clean_inject_BayesNet, testData));

		String targetOnlyModel = new String(targetModelLoc+targetLearnedModelName);
		BIFReader targetOnlyModel_BayesNet = new BIFReader();
		targetOnlyModel_BayesNet.processFile(targetOnlyModel);
	//	System.out.print(targetOnlyModel_BayesNet.graph());
		System.out.println("AUC_target_only:" + evaluateBN(targetOnlyModel_BayesNet, testData));

		String trueTargetModel = new String(targetModelLoc+targetTrueModelName);
		BIFReader ture_target_BayesNet = new BIFReader();
		ture_target_BayesNet.processFile(trueTargetModel);
		System.out.println("AUC_true_target:" + evaluateBN(ture_target_BayesNet, testData));

		*/

    } // buildStructure


    /**
     * Assigns a so-called "prequential" Bayesian score to the model.
     * This scores captures how well the model predicts the target node.
     * In particular, it represents the probability of the target node data, given
     * the model and training data. It is a conditional, marginal likelihood.
     *
     * @param node is the target node
     * @return score
     */
    protected double scoreNode(int node) {
        double lnTotalScore = 0.0;

        if (parents[node][0] > 0) {
            byte firstParentSize = nodeDimension[parents[node][1]];

            for (int i = 1; i <= firstParentSize; i++)
                countsTree[i] = 0;

            countsTreePtr = firstParentSize + 1;
            countsPtr = 1;
        } else {
            countsTreePtr = 1;
            countsPtr = nodeDimension[node] + 1;

            for (int i = 1; i <= nodeDimension[node]; i++)
                counts[i] = 0;
        }


        // this builds a frequency tree
        for (int casei = firstCase; casei <= lastCase; casei++)
            fileCase(node, casei);

        // zero the counts in the frequency tree, but keep the tree
        for (int i = 1; i <= countsPtr; i++)
            counts[i] = 0;

        double a= 1.0;
        double b=  (double) nodeDimension[node];  //default: K2 scoring

        //if (getScoreMetric().getSelectedTag().getIDStr().equals("0")) {  // K2 scoring measure
        if (m_ScoreMetric==1) {  // BDeu scoring measure
            a = m_SourceSampleSize / numberOfJointStates(node);
            b = a * (double) nodeDimension[node];
        }

        String nodeName = nodeInfo[node].name;
        //System.out.println("a: "+a+"\tb: "+b);

        // derive the conditional prequential score
        for (int casei = firstCase; casei <= lastCase; casei++) {
            if (m_ScoreMetric==0 | m_ScoreMetric==1 | !sharedNodeList.contains(nodeName)) {
                deriveNodeProbs(casei, node, a, b);
            }
            else if (m_ScoreMetric==2 & sharedNodeList.contains(nodeName)) {
                //	System.out.println("derive node prob using prior model and target data");
                double tempAdjustPriorSampleSize = adjustedPriorSampleSize;
                if (m_AdjustSourceSampleSizeMetric == 3 | m_AdjustSourceSampleSizeMetric == 6 | m_AdjustSourceSampleSizeMetric == 7){
                    //if it is shared feature, then pesudocount is weighted by bayes factor
                    tempAdjustPriorSampleSize = weightTable.get(nodeName) * adjustedPriorSampleSize;
                    //	System.out.println("tempSampleSizeFor " + nodeName + tempAdjustPriorSampleSize)
                }
                deriveNodeProbs_Prior_Data(casei, node,tempAdjustPriorSampleSize);
            }
            byte nodeValue_casei = cases[casei][node];
            lnTotalProb = MINUS_INFINITY;
            //System.out.println("node: "+node);
            //System.out.println("lnTotalProb = "+lnTotalProb+"\tnodeValue_casei = "+nodeValue_casei+"\tnodeProb["+node+"] = "+nodeProb[node-1]);

            double lnNumer = 0;
            for (int j = 1; j <= nodeDimension[node]; j++) {
                //System.out.println("nodeProb["+j+"] = "+ nodeProb[j] + "  --> lnChildrenProb["+casei+"]["+j+"] = "+lnChildrenProb[casei][j]);
                //	System.out.println(nodeProb[j]);
                //	System.out.println(lnChildrenProb[casei][j]);
                double lnMarginal = Math.log(nodeProb[j]) + lnChildrenProb[casei][j];
                lnTotalProb = lnXpluslnY(lnTotalProb, lnMarginal); //lnTotalProb = ln (TotalProb + marginal)
                if (j == nodeValue_casei)
                    lnNumer = lnMarginal;
                if (print && casei==lastCase){
                    System.out.println("node" + node + "--nodeProb[j]-"+nodeProb[j]+ "...lnChildrenProb[casei][j]-" + lnChildrenProb[casei][j]);
                }

            }

            double lnNodeProbAll = lnNumer - lnTotalProb;
            //System.out.println(lnNodeProbAll);
            // lnNodeProbAll is the prob of the node value of case_i give all the predictors,
            // both parent predictors and children predictors.
            lnTotalScore += lnNodeProbAll;
            fileCase(node, casei);
        }
        //	System.out.println("current score:" + lnTotalScore);
        return lnTotalScore;
    }

    /**
     * This function is similar to ScoreNode, except it just incorporates
     * the contribution for node to lnChildrenProb.
     *
     * @param node
     */
    protected void incorporateNodeScores(int node) {
        initializeFileCase(node);
        // this builds a frequency tree
        for (int casei = firstCase; casei <= lastCase; casei++)
            fileCase(node, casei);
        // zero the counts in the frequency tree, but keep the tree
        for (int i = 1; i <= countsPtr; i++)
            counts[i] = 0;
        double a=1.0;
        double b=(double) nodeDimension[node];
        //if (getScoreMetric().getSelectedTag().getIDStr().equals("0")) { //K2 is selected
        if (m_ScoreMetric==1){
            a = m_SourceSampleSize / numberOfJointStates(node);
            b = a * (double) nodeDimension[node];
        }
        String nodeName = nodeInfo[node].name;
        // derive the conditional probabilities of node
        for (int casei = firstCase; casei <= lastCase; casei++) {
            byte saveTargetValue = cases[casei][targetNode];
            int nodeValue_casei = cases[casei][node];
            for (int targetValue = 1; targetValue <= nodeDimension[targetNode]; targetValue++) {
                cases[casei][targetNode] = (byte) targetValue;
                if (m_ScoreMetric==0 | m_ScoreMetric==1 | !sharedNodeList.contains(nodeName)) {
                    deriveNodeProbs(casei, node, a, b);
                }
                else if (m_ScoreMetric==2 && sharedNodeList.contains(nodeName)) {
                    //for other m_AdjustSourceSampleSizeMetric method
                    double tempAdjustPriorSampleSize = adjustedPriorSampleSize;
                    if (m_AdjustSourceSampleSizeMetric == 3 | m_AdjustSourceSampleSizeMetric == 6 | m_AdjustSourceSampleSizeMetric == 7){
                        //if it is shared feature, then pesudocount is weighted by bayes factor
                        tempAdjustPriorSampleSize = weightTable.get(nodeName) * adjustedPriorSampleSize;
                        //System.out.println("tempSampleSizeFor " + nodeName + tempAdjustPriorSampleSize);
                    }
                    deriveNodeProbs_Prior_Data(casei, node,tempAdjustPriorSampleSize);
                }

                double lnProb = Math.log(nodeProb[nodeValue_casei]);

                lnChildrenProb[casei][targetValue] += lnProb;
                lnChildProb[casei][targetValue][map[node]] = lnProb;
            }

            cases[casei][targetNode] = saveTargetValue;
            fileCase(node, casei);
        }
    }


    public void setConfig() throws FileNotFoundException{
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

        featureSelectionApproach = Utility.getConfig("featureSelectionApproach",configFileName);
        modelLearningApproach = Utility.getConfig("modelLearningApproach",configFileName);

        sourceLearnedModelName=sourceDataName.replace(".arff", "-" + featureSelectionApproach + "-" + modelLearningApproach + ".bif");
        sourceLearnedCleanedModelName = sourceLearnedModelName.replace(".bif", "")+"_cleaned.bif";
        targetLearnedModelName=targetDataName.replace(".arff", "-" + featureSelectionApproach + "-" + modelLearningApproach + ".bif");
        targetFinalModelName=sourceLearnedModelName.replace(".bif","")+"-"+targetDataName.replace(".arff", "")
                + "-"+ transferLearningApproach + "-" + transferLearningWeight + ".bif";

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

    public void adjustSourceSampleSize() throws Exception {
        double weight = 1.0;
        if (m_AdjustSourceSampleSizeMetric == 0){
            weight = 1.0;
        }
        else if (m_AdjustSourceSampleSizeMetric == 1) // ratio approach
        {
            int targetSampleSize = targetInstances.numInstances();
            //	System.out.println("target size: " + targetSampleSize);
            //	System.out.println("source size: " + m_SourceSampleSize);
            if (targetSampleSize < m_SourceSampleSize){
                weight = 1.0* targetSampleSize / m_SourceSampleSize;
            }
            else weight = 1.0;
        }
        else if (m_AdjustSourceSampleSizeMetric == 2) // KL approach
        {
            double KL = 1.0;
            if (useTrueSourceDataForWeight){
                KL = trueKL ;
            }
            else {
                KL = avgKL;
            }
            weight = Math.pow(2, -KL);
            System.out.println("weight:"+weight);
        }
        else if (m_AdjustSourceSampleSizeMetric == 4) // new KL approach using learned source model
        {
            double KL = KL_targetData_learnedSourceModel;
            weight = Math.pow(2, -KL);
            System.out.println("weight:"+weight);
        }
        else if (m_AdjustSourceSampleSizeMetric == 21) // adjust new KL approach using true source model
        {
            double KL = KL_targetData_learnedSourceModel;
            weight = Math.pow(2, -KL)/targetInstances.numInstances();
            System.out.println("weight:"+weight);
        }
        else if (m_AdjustSourceSampleSizeMetric == 5) // new KL approach using true source model
        {
            double KL = KL_targetData_trueSourceModel;
            weight = Math.pow(2, -KL);
            System.out.println("weight:"+weight);
        }
        else if (m_AdjustSourceSampleSizeMetric == 20) // adjust new KL approach using true source model
        {
            double KL = KL_targetData_trueSourceModel;
            weight = Math.pow(2, -KL)/targetInstances.numInstances();
            System.out.println("weight:"+weight);
        }
        else if (m_AdjustSourceSampleSizeMetric == 3) // Bayes factor approach
        {
            ArffLoader arff2 = new ArffLoader();
            arff2.setFile(new File(sourceDataLoc+sourceDataName));
            sourceInstances = arff2.getDataSet();
            if (useTrueSourceDataForWeight){
                ArffLoader arff1 = new ArffLoader();
                arff1.setFile(new File(targetDataLoc+targetTestDataName));
                Instances  targetTestInstances = arff1.getDataSet();
//				weightTable = Utility.getTrueAverageBayesFactorTable(targetNodeName, sharedNodeList,
//						sourceDataSize, sourceModelLoc,sourceTrueModelName,
//						targetModelLoc, targetTrueModelName,sourceSimulateDataLoc) ;
                weightTable = trueBFWeightTable ;
            }
            else {
//				weightTable = Utility.getAverageBayesFactorTable(targetNodeName, sharedNodeList,
//					targetInstances,sourceDataSize,sourceModelLoc,sourceLearnedCleanedModelName,sourceSimulateDataLoc,
//					sourceDataLoc, sourceDataName);
                weightTable = avgBFWeightTable ;
            }
            for (int i=0; i<sharedNodeList.size(); i++){
                String key = sharedNodeList.get(i);
                System.out.println(key+","+weightTable.get(key));
            }
            System.out.println("Saved weightTable, but did not change adjustedPriorSampleSize here for individual node!");
        }
        else if (m_AdjustSourceSampleSizeMetric == 6) // new KL for each nodes. Use target data and learned source model
        {
            weightTable = new HashMap<String,Double>();
            for (int i=0; i<sharedNodeList.size(); i++){
                String key = sharedNodeList.get(i);
                Double oneKL = learnedSource_nodeKLTable.get(key);
                Double oneWeight = Math.pow(2, -oneKL);
                weightTable.put(key, oneWeight);
                System.out.println(key+","+weightTable.get(key));
            }
            System.out.println("Saved weightTable, but did not change adjustedPriorSampleSize here for individual node!");
        }
        else if (m_AdjustSourceSampleSizeMetric == 7) // new KL for each nodes. Use target data and true source model
        {
            weightTable = new HashMap<String,Double>();
            for (int i=0; i<sharedNodeList.size(); i++){
                String key = sharedNodeList.get(i);
                System.out.println(key);
                Double oneKL = trueSource_nodeKLTable.get(key);
                Double oneWeight = Math.pow(2, -oneKL);
                weightTable.put(key, oneWeight);
                System.out.println(key+","+weightTable.get(key));
            }
            System.out.println("Saved weightTable, but did not change adjustedPriorSampleSize here for individual node!");
        }
        adjustedPriorSampleSize = weight * m_SourceSampleSize;
    }




    public void getCPTCombineModelData() throws Exception{
        SimpleEstimator cptEstimator = new SimpleEstimator();
        cptEstimator.setAlpha(0.0);
        m_BayesNet.setEstimator(cptEstimator);
        m_BayesNet.estimateCPTs();
        //buildClassifier has estimateCPT() after buildStructure()
        /*get pesudo counts, get adjusted distributions, set distributions for the final Bayesian network*/
        Estimator[][] estimators = m_BayesNet.getDistributions();
        System.out.println("-------------------Printing counts----------------");
        for (int iNode=0; iNode<estimators.length; iNode++){
            String iNodeName = m_BayesNet.getNodeName(iNode);
            System.out.print("Node " + iNodeName + ":");
            ParentSet theParents = m_BayesNet.getParentSet(iNode);
            for (int parentIndex=0; parentIndex<theParents.getNrOfParents(); parentIndex++){
                System.out.println(m_BayesNet.getNodeName(theParents.getParent(parentIndex)));
            }

            //Get all permutations of parents of iNode (first parent, second parent, PP, PA, AP, AA)
            ArrayList<ArrayList<String>> permutationList = getAllPermutation(iNode);
            //P(the node, the first parent, the second parent in prior network)

            Map<List<String>,Double> priorProbTable = new HashMap<List<String>,Double>();
            if (sharedNodeList.contains(iNodeName)) {
                priorProbTable = getPriorJointProbTable(iNode);
            }

            int nNodeValues = m_BayesNet.getCardinality(iNode);
            int nParentConfig = theParents.getCardinalityOfParents();
            if (nParentConfig!=permutationList.size()){
                throw new Exception("parents' number of configurations do not match!");
            }
            //double[][] probs = new double[nParentConfig][nNodeValues];
            for (int jParentsConfig=0; jParentsConfig<nParentConfig; jParentsConfig++){
                ArrayList<String> oneParentConfig =  permutationList.get(jParentsConfig);
                DiscreteEstimatorBayes theCount = (DiscreteEstimatorBayes) estimators[iNode][jParentsConfig];
                for (int iNodeValue=0; iNodeValue<nNodeValues; iNodeValue++){
                    String iNodeValueString = m_BayesNet.getNodeValue(iNode, iNodeValue);
                    ArrayList<String> temp = new ArrayList<String> (oneParentConfig);
                    oneParentConfig.add(0, iNodeValueString); //Now, it is one config of iNode and its parents
                    System.out.println();
                    for (int k=0; k<oneParentConfig.size(); k++){
                        System.out.print(oneParentConfig.get(k));
                    }
                    double data =1.0*iNodeValue;

                    System.out.print("count in data:" + theCount);
                    //for target only features, add 1 to avoid 0 probability
                    if (!sharedNodeList.contains(iNodeName)){
                        theCount.addValue(data, new Double("1.0"));
                    }
                    else //for shared features
                    {
                        //for other m_AdjustSourceSampleSizeMetric approach
                        double thePesudoCount = adjustedPriorSampleSize*priorProbTable.get(oneParentConfig);
                        if (m_AdjustSourceSampleSizeMetric==3 | m_AdjustSourceSampleSizeMetric==6 | m_AdjustSourceSampleSizeMetric==7){
                            thePesudoCount = thePesudoCount*weightTable.get(iNodeName);
                        }
                        theCount.addValue(data, thePesudoCount);
                        System.out.print("pesudo counts:" + thePesudoCount);
                    }
                    oneParentConfig = temp; //remove the iNodeValue for another iNodeValue in the for loop
                }
                System.out.print(theCount);
            } // end for loop for one parent config
            System.out.println("");

        }//end for loop
    }

    /**
     * Return parents permutations of iNode
     * @param iNode
     * @return
     */
    private ArrayList<ArrayList<String>> getAllPermutation(int iNode){
        ArrayList<Integer> nodeList = new ArrayList<Integer>();
        ParentSet theParents = m_BayesNet.getParentSet(iNode);
        for (int parentIndex=0; parentIndex<theParents.getNrOfParents(); parentIndex++){
            nodeList.add(theParents.getParent(parentIndex));
        }
        ArrayList<String> theConfig = new ArrayList<String>();
        ArrayList<ArrayList<String>> configList = new ArrayList<ArrayList<String>>();
        getOnePermutation(theConfig, nodeList,0,configList);
        return configList;
    }

    /**
     * Return parents permutations of iNode
     * @param iNode
     * @return
     */
    private ArrayList<ArrayList<String>> getAllPermutationIncludeNode(int iNode){
        ArrayList<Integer> nodeList = new ArrayList<Integer>();
        nodeList.add(iNode);
        ParentSet theParents = m_BayesNet.getParentSet(iNode);
        for (int parentIndex=0; parentIndex<theParents.getNrOfParents(); parentIndex++){
            nodeList.add(theParents.getParent(parentIndex));
        }
        ArrayList<String> theConfig = new ArrayList<String>();
        ArrayList<ArrayList<String>> configList = new ArrayList<ArrayList<String>>();
        getOnePermutation(theConfig, nodeList,0,configList);
        return configList;
    }





    private void getOnePermutation(ArrayList<String> theConfig,
                                   ArrayList<Integer> nodeList, int currentIndex, ArrayList<ArrayList<String>> configList){
        if (currentIndex == nodeList.size()) {
            configList.add(theConfig);
            return;
        }
        int iNode = nodeList.get(currentIndex);
        for (int iValue=0; iValue<m_BayesNet.getCardinality(iNode);iValue++){
            String theValue = m_BayesNet.getNodeValue(iNode, iValue);
            ArrayList<String> lastConfig = new ArrayList(theConfig);
            theConfig.add(theValue);
            getOnePermutation(theConfig,nodeList,currentIndex+1,configList);
            theConfig = lastConfig;
        }
    }

    /**
     * Return permutations of nodeNameList
     * @param iNode
     * @return
     */
    private ArrayList<ArrayList<String>> getAllPermutationForList(ArrayList<String> nodeNameList){
        ArrayList<Integer> nodeList = new ArrayList<Integer>();
        for (int iNode=0; iNode<nodeNameList.size(); iNode++){
            String oneNode = nodeNameList.get(iNode);
            int oneNodeIndex = nameIndexTable.get(oneNode);
            nodeList.add(oneNodeIndex);
        }
        ArrayList<String> theConfig = new ArrayList<String>();
        ArrayList<ArrayList<String>> configList = new ArrayList<ArrayList<String>>();
        getOnePermutation(theConfig, nodeList,0,configList);
        return configList;
    }



    /**
     * Return a table containing P(the node, the first parent, the second parent in prior network)
     * @param iNode
     * @param oneAdjustedPriorSampleSize
     * @return
     */
    public Map<List<String>,Double> getPriorJointProbTable(int iNode){
        ArrayList<ArrayList<String>> permutationList = getAllPermutationIncludeNode(iNode);
        ArrayList<String> nodeNameList = new ArrayList<String>();
        nodeNameList.add(m_BayesNet.getNodeName(iNode));
        ParentSet theParents = m_BayesNet.getParentSet(iNode);
        for (int parentIndex=0; parentIndex<theParents.getNrOfParents(); parentIndex++){
            nodeNameList.add(m_BayesNet.getNodeName(theParents.getParent(parentIndex)));
        }
//		for (int k=0; k<nodeNameList.size(); k++){
//			System.out.print(nodeNameList.get(k));
//		}
//		System.out.println();
        Map<List<String>,Double> jointProbs = smileNet.probabilities(nodeNameList, permutationList);
        Set<List<String>> keySet = jointProbs.keySet();
//		for (List<String> oneList : keySet){
//			for (int i=0; i<oneList.size(); i++){
//				System.out.print(oneList.get(i));
//			}
//			System.out.println(jointProbs.get(oneList));
//		}
        return jointProbs;
    }





    public void searchParentsEBMC1(int node) {
        boolean allTrue;
        double bestScore, score;
        int bestParent, parent;
        parents[node][0] = 0; //delete this line??
        //  System.out.println(String.format("=========== score: %f", scoreNode(node)));
        if (m_ScoreMetric==0 | m_ScoreMetric==1){
            nodeScore = scoreNode(node) + prior_EBMC1(node);
        }
        else  nodeScore = scoreNode(node) + logStructurePrior();
        numberOfModelsScored++;
        System.out.println(String.format("current score: %10.5f time: %11.4f mins", nodeScore, timeMinutes()));
        while (parents[node][0] < maxParents) {
            bestScore = 1;
            bestParent = 0;
            allTrue = true;
            System.out.println(String.format("starting search for an additional predictor at %11.4f mins ...", timeMinutes()));
            // The last conjunct above insures that the first parent of the target will  not already be a child of the target.
            // This condition guarantees that the predictors (parents) of the target will be a unique set.
            // Otherwise, the search could get stuck in an infinite loop.
            for (int i = 1; i <= numberOfNodes; i++) {
                if ((i != node) && !parentPresent[i] && !childPresent[i]) {
                    allTrue = false;
                    parent = i;
                    parents[node][0]++;
                    parents[node][parents[node][0]] = parent;
                    if (m_ScoreMetric==0 | m_ScoreMetric==1){
                        score = scoreNode(node) + prior_EBMC1(node);
                    }
                    else  score = scoreNode(node) + logStructurePrior();
                    numberOfModelsScored++;
                    parents[node][0]--;

                    //    System.out.println("temporary score: " + score);
                    if ((score > bestScore) || (bestScore == 1)) {
                        bestParent = parent;
                        bestScore = score;
                    }
                }
            }

            if (allTrue) {
                break;
            }

            if (bestScore > nodeScore) {
                nodeScore = bestScore;
                parentPresent[bestParent] = true;
                parents[node][0]++;
                parents[node][parents[node][0]] = bestParent;
                StringBuilder info = new StringBuilder("current new predictors: ");
                for (int j = 1; j <= parents[node][0]; j++) {
                    info.append(String.format("%s ", nodeInfo[parents[node][j]].name)); //modify nodeInfo part
                }
                info.append(String.format(" score: %10.5f time: %11.4f mins", nodeScore, timeMinutes()));
                System.out.println(info.toString());
            } else {
                break;
            }
        }  // end of while-loop

        // reset to all false for future uses
        for (int i = 1; i <= parents[node][0]; i++) {
            parentPresent[parents[node][i]] = false;
        }
    }


    /**
     * remove a node from source model when the node does not appear in target data
     * @param bayesNet
     * @param targetInstances
     * @return
     * @throws Exception
     */
    public BayesNet cleanSourceBN(BayesNet bayesNet, Instances targetInstances) throws Exception{
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

    public void getSharedList(BayesNet bayesNet) throws Exception{
        sharedNodeList = new ArrayList<String>();
        int numNodes= bayesNet.getNrOfNodes();
        for (int iNode=0; iNode<numNodes; iNode++){
            String nodeName =  bayesNet.getNodeName(iNode);
            sharedNodeList.add(nodeName);
        }
    }


    /**
     * inject target specific features into a bayesnetwork, use prevalence in target data as prob.
     * @param bayesNet
     * @param data
     * @return
     * @throws Exception
     */
    public BayesNet injectTargetSpecificFeature(BayesNet bayesNet, Instances data) throws Exception{
        System.out.println("------Injecting target specific features");
        PrintWriter printout = new PrintWriter(new File(utilityLoc+temporaryFileName));
        printout.println(bayesNet.graph());
        printout.flush(); printout.close();
        BIFReader reader = new BIFReader();
        EditableBayesNet tempNet = new EditableBayesNet(reader.processFile(utilityLoc+temporaryFileName));
        ArrayList<String> nodeList = new ArrayList<String>();
        int numNodes= tempNet.getNrOfNodes();
        for (int iNode=0; iNode<numNodes; iNode++){
            String nodeName =  tempNet.getNodeName(iNode);
            if (!nodeList.contains(nodeName)){
                nodeList.add(nodeName);
            }
        }
        //get a prob table from target data
        HashMap<String, HashMap<String,Double>> table = getProb(data);
        String targetNodeName = data.attribute(data.numAttributes()-1).name();
        int numAttribute = data.numAttributes();
        for (int i=0; i<numAttribute; i++){
            Attribute oneAttribute = data.attribute(i);
            String candidateFeature = oneAttribute.name();

            //If a target feature is not in node list, then inject it into the bayesian network
            //change name, use the prob in target data as CPT for injected features.
            if (!nodeList.contains(candidateFeature)){
                tempNet.addNode(candidateFeature, oneAttribute.numValues());
                tempNet.addArc(tempNet.getNode(targetNodeName), tempNet.getNode(candidateFeature)); //add link, will make it dummy
                int numberValueTargetNode = data.numClasses();
                HashMap<String,Double> attributeProbTable = table.get(candidateFeature);
                String[] valueList = attributeProbTable.keySet().toArray(new String[0]);
                double[][] prob = new double[numberValueTargetNode][valueList.length];
                //make it dummy here
                for (int k=0; k<numberValueTargetNode; k++){
                    for (int j=0; j<valueList.length; j++){
                        String name = valueList[j];
                        int num = j+1;
                        String oldName = "Value" + num;
                        tempNet.renameNodeValue(tempNet.getNode(candidateFeature), oldName, name);
                        prob[k][j] = attributeProbTable.get(name);
                    }
                }
                tempNet.setDistribution(candidateFeature, prob);
                System.out.println("inject:" + candidateFeature);
            }
        }
        PrintWriter printout2 = new PrintWriter(new File(sourceModelLoc+sourceCleanInjectModelXML));
        printout2.println(tempNet.graph());
        printout2.flush(); printout2.close();
        ConvertBIFtoGENIE conv = new ConvertBIFtoGENIE();
        conv.runner(sourceModelLoc+sourceCleanInjectModelXML, "", utilityLoc+hashCodeName, "xdsl");
        return tempNet;
    }

    public EditableBayesNet copyBN(BayesNet bayesNet) throws Exception{
        PrintWriter printout = new PrintWriter(new File(utilityLoc+temporaryFileName));
        printout.println(bayesNet.graph());
        printout.flush(); printout.close();
        BIFReader reader = new BIFReader();
        EditableBayesNet copiedNet  = new EditableBayesNet(reader.processFile(utilityLoc+temporaryFileName));
        copiedNet.m_Instances = bayesNet.m_Instances;
        return copiedNet;
    }
    /**
     * return a table list each feature and its prob distribution
     * @param data
     * @return
     */
    public HashMap<String, HashMap<String,Double>> getProb(Instances data){
        HashMap<String, HashMap<String,Double>> table = new HashMap<String, HashMap<String,Double>>();
        int numAttribute = data.numAttributes();
        for (int i=0; i<numAttribute; i++){
            Attribute oneAttribute = data.attribute(i);
            String nameAttribute = oneAttribute.name();
            AttributeStats stats = data.attributeStats(i);
            int total = Utils.sum(stats.nominalCounts) + oneAttribute.numValues();
            HashMap<String,Double> tableForAttribute = new HashMap<String,Double>();
            Double remain=1.0;
            for (int j = 0; j < oneAttribute.numValues()-1; j++){
                String value = oneAttribute.value(j);
                //keep 4 decimals
                Double temp = (double) (stats.nominalCounts[j]+1)/(double) total;
                Double prob =  Double.parseDouble(String.format("%.4g%n",temp));
                tableForAttribute.put(value, prob);
                remain = remain - prob;
            }
            String value = oneAttribute.value(oneAttribute.numValues()-1);
            Double prob =  Double.parseDouble(String.format("%.4g%n", remain));
            tableForAttribute.put(value, prob);
            table.put(nameAttribute,tableForAttribute);
        }

        return table;
    }

    private String printOneCase(int casei){
        String toPrint = "";
        for (int iNode=1; iNode<=numberOfNodes; iNode++){
            toPrint = toPrint + cases[casei][iNode] + ",";
        }
        return toPrint;
    }

    /**
     * This method calculates Derives the probability distribution over the target node
     * given the values of its parents in casei. The cases from
     * 1 to casei-1 are used to parameterize this predictive distribution.
     * It is revised from deriveNodeProbs method.
     * Nijk' = N' * p(Xi=k, Pai=j | Bsc) by calling the getPriorProb method.
     * Nij' = sum of Nijk'
     * @param casei
     * @param node
     * @param priorNetwork
     * @param oneAdjustedPriorSampleSize
     */
    protected void deriveNodeProbs_Prior_Data(int casei, int node, double oneAdjustedPriorSampleSize) {
        if (casei==lastCase){
            //  	System.out.println("derive prob for " + nodeInfo[node].name + " prior sample size: " + oneAdjustedPriorSampleSize);
        }
        int numberOfParents, parentValue;
        int ctPtr, cPtr, ptr;
        double[] k = new double[nodeDimension[node]+1]; //an array to save Nijk
        double v=0.0; //Nij
        numberOfParents = parents[node][0];
        if (casei==lastCase){
            //	System.out.println("number of parents:" + numberOfParents);
        }
        //Ye note This array saves the values of node's parents in case i
        for (int i = 1; i <= numberOfParents; i++)
            values[i] = cases[casei][parents[node][i]];

        //calculate Nij first
        for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++){
            //	System.out.println(nodeInfo[node].name);
            k[nodeValue] = 1.0*oneAdjustedPriorSampleSize*getPriorProb(node, nodeValue, values);
            v += k[nodeValue];

        }

        ctPtr = 1;
        for (int i = 1; i <= numberOfParents; i++) {
            parentValue = values[i];
            ptr = countsTree[ctPtr + parentValue - 1]; //Ye note Here, it already puts the parent info in case i into consideration.
            if (ptr > 0)
                ctPtr = ptr;
            else {  // there are no previous cases that match the current parent values of node, so return a uniform distribution
                for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++){
                    nodeProb[nodeValue] = 1.0* k[nodeValue] / v;
                }
                return;
            }
        }
        cPtr = ctPtr;

        double b = 0;
        for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++){
            b += counts[cPtr + nodeValue - 1];  //b = Nij
        }

        for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++) {
            double a = counts[cPtr + nodeValue - 1]; // a=Nijk
            // nodeProb[nodeValue] = P(node = nodeValue | parents_node, case1,...,casei-1)
            nodeProb[nodeValue] = (a + k[nodeValue]) / (b + v);

            if (casei==lastCase){
                //System.out.println("casei: " + casei + " " + printOneCase(casei) + " node " + nodeInfo[node].name);
                //System.out.println("nodeProb[" + nodeValue+ "]" + nodeProb[nodeValue]+" count:" + a + " pesudoCount"
                //	+ k[nodeValue] + " totalCount:" + b + " totalPesudoCount " + v );
            }


        }
    }


    /**
     * This method is called by deriveNodeProbs_Prior_Data and it is used for score Node.
     * p(node=nodeValue, parents = parentValues | prior model)
     * @param node
     * @param nodeValue
     * @param parentValues
     * @return
     */
    public double getPriorProb(int node, int nodeValue, int[] parentValues){
//    		System.out.print(nodeInfo[node].name + "=" + nodeInfo[node].value[nodeValue] + ",parents:" );
//	    	for (int i=1; i<=parents[node][0]; i++){
//	    		int parent = parents[node][i];
//	    		System.out.print(nodeInfo[parent].name+ " =" + nodeInfo[parent].value[parentValues[i]] +",");
//	    	}
//	    	System.out.println();

        int numberOfParents = parents[node][0];
        Double prob = new Double(new String("1.0"));
        ArrayList<String> nodeNameList = new ArrayList<String>();
        ArrayList<String> currentAssignment = new ArrayList<String>();
        //Ye Node: kevin code node number is 1 index more than the node number in the weka Bayesian.
        //Think about whether I should add 1 for the node later.
        //nodeInfo is variable in Kevin's code. Do not need to deal with plus or minus 1 issue.
        nodeNameList.add(new String(nodeInfo[node].name));
        currentAssignment.add(new String(""+nodeInfo[node].value[nodeValue]));
        //System.out.println("nodeValue"+nodeInfo[node].value[nodeValue]);
        for (int k=1; k<=numberOfParents; k++){
            String parentName = nodeInfo[parents[node][k]].name;
            nodeNameList.add(parentName);
            int parentValue = parentValues[k];
            String parentValueString = nodeInfo[parents[node][k]].value[parentValue];
            //	System.out.println(parentValueString);
            currentAssignment.add(new String(parentValueString));
        }

        ArrayList<ArrayList<String>> pertumationList = getAllPermutationForList(nodeNameList);
//	    	System.out.println("nodeList");
//	    	for (int p=0; p<nodeNameList.size(); p++){
//	    		System.out.println(nodeNameList.get(p));
//	    	}
//	    	System.out.println("pertumationList");
//	    	for (int p=0; p<pertumationList.size(); p++){
//	    		ArrayList<String> oneArray = pertumationList.get(p);
//	    		for (int k=0; k<oneArray.size(); k++){
//	    			System.out.print(oneArray.get(k)+",");
//	    		}
//	    		System.out.println();
//	    	}
//	    	System.out.println("currentAssignment");
//	    	for (int p=0; p<currentAssignment.size(); p++){
//	    		System.out.print(currentAssignment.get(p)+",");
//	    	}
//	    	System.out.println();
        Map<List<String>,Double> conditionalProbs = smileNet.probabilities(nodeNameList, pertumationList);
        prob = conditionalProbs.get(currentAssignment);
        Set<List<String>> keySet = conditionalProbs.keySet(); //the keySet lists all possible configurations
        //if (parentValues.length==2){
//				for (List<String> oneList : keySet){
//					for (int i=0; i<oneList.size(); i++){
//						System.out.print(oneList.get(i));
//					}
//					System.out.println(conditionalProbs.get(oneList));
//				}
//				System.out.println("currentAssisgment:");
//				for (int s=0; s<currentAssignment.size(); s++){
//					System.out.print(currentAssignment.get(s));
//				}
//				System.out.println(prob+"");
        //}


        return prob;
    }


    public double logStructurePrior() {
        int[][] convertedModel = getConvertedCandiateModel();
//	    	System.out.println("****current candidate:");
//	    	System.out.println(printArray(convertedModel));
//	    	System.out.println(source_clean_BayesNet);
        double k = 1.0/(1+ adjustedPriorSampleSize);
        int sizeParentDiff = 0;
        //	System.out.println("m_PriorEquivalentSampleSize "+m_PriorEquivalentSampleSize);
        //Save names of nodes in source_clean_BayesNet
        ArrayList<String> nodesInSourceClean = new ArrayList<String>();
        //For source_clean_BayesNet, this table saves a node's name and its index
        HashMap<String,Integer> sourceNodesNameIndexTable = new HashMap<String,Integer>();
        Instances sourceCleanInstances = source_clean_BayesNet.m_Instances;
        for (int jNode=0; jNode<sourceCleanInstances.numAttributes(); jNode++){
            String jNodeName = sourceCleanInstances.attribute(jNode).name();
            nodesInSourceClean.add(jNodeName);
            sourceNodesNameIndexTable.put(jNodeName, jNode);
        }
        double logStructureScore = 0.0;
        for (int iNode=1; iNode<=numberOfNodes; iNode++){
            String iNodeName = nodeInfo[iNode].name;
            //	System.out.println("iNode:"+iNode+" "+iNodeName);

            //only penalize features that are also in source. If not, do not penalize the structure difference.
            //nodesInSourceClean should equal to sharedNodeList. Order of nodes may be different, but the content should be the same.
            if (nodesInSourceClean.contains(iNodeName)){
                //		System.out.println("iNode:"+iNode+" "+iNodeName);
                List<String> parentSetInSource = new ArrayList<String>();
                int iNodeIndexInSource = sourceNodesNameIndexTable.get(iNodeName);
                //		System.out.println(iNodeIndexInSource+"");
                ParentSet temp = source_clean_BayesNet.getParentSet(iNodeIndexInSource);
                //		System.out.println("number of parents:"+temp.getNrOfParents());
                int[] parentsInSourceClean = temp.getParents();
                //		System.out.println("parents in source clean:");
                for (int i=0; i<temp.getNrOfParents(); i++){
                    int indexParentInSource = parentsInSourceClean[i];
                    //			System.out.println("parent:"+sourceCleanInstances.attribute(indexParentInSource).name());
                    parentSetInSource.add(sourceCleanInstances.attribute(indexParentInSource).name());
                }
                //		System.out.println("parents in candidate:");
                List<String> parentSetInCandiate = new ArrayList<String>();
                int[] parentsInCandidate = convertedModel[iNode];
                int numberOfParentInCandidate = parentsInCandidate[0];
                //		System.out.println("number of parents:"+numberOfParentInCandidate);
                for (int i=1; i<=numberOfParentInCandidate; i++){
                    int indexParentInCandidate = parentsInCandidate[i];
                    //			System.out.println(indexParentInCandidate+"");
                    String theParent = nodeInfo[indexParentInCandidate].name;
                    //			System.out.println("		" + theParent);
                    if (nodesInSourceClean.contains(theParent)){
                        parentSetInCandiate.add(theParent);
                        //				System.out.println("		add" + theParent);
                    }
                }
                //		System.out.println("parent set in source:");
                //		printList(parentSetInSource);
                //		System.out.println("parent set in candidate:");
                //		printList(parentSetInCandiate);
                int sizeParentDiff_Nodei  = difference(parentSetInSource, parentSetInCandiate);
                //		System.out.println("difference " + difference(parentSetInSource, parentSetInCandiate));
                sizeParentDiff  = sizeParentDiff + sizeParentDiff_Nodei;
                if (m_AdjustSourceSampleSizeMetric == 3 | m_AdjustSourceSampleSizeMetric == 6 | m_AdjustSourceSampleSizeMetric == 7 ) // for bayes factor
                //if it is shared feature, then pesudocount is weighted by bayes factor
                {
                    if (sharedNodeList.contains(iNodeName)) {
                        double k_Nodei = 1.0/(1+ adjustedPriorSampleSize*weightTable.get(iNodeName));
                        //System.out.println("iNodeName " + thePesudoCount);
                        logStructureScore = logStructureScore + sizeParentDiff_Nodei * Math.log(k_Nodei);
                        //	System.out.println(iNodeName + " k_Nodei " + k_Nodei);
                        //	System.out.println(logStructureScore);
                    }
                    //if not contain Key, then it is target specific feature and the difference is not counted.
                }
                else if (m_AdjustSourceSampleSizeMetric <3 | m_AdjustSourceSampleSizeMetric==4 | m_AdjustSourceSampleSizeMetric==5
                        | m_AdjustSourceSampleSizeMetric==20 | m_AdjustSourceSampleSizeMetric==21){
                    //	System.out.println("logStructureScore:" + logStructureScore );
                    logStructureScore = logStructureScore + sizeParentDiff_Nodei * Math.log(k);
                    //	System.out.println("logStructureScore:" + logStructureScore );
                }
            }
        }
        //    	System.out.println("K:" + k);
        //   	System.out.println("sizeParentDiff:" + sizeParentDiff);
        //   	System.out.println("structure score:" + logStructureScore);
        return logStructureScore ;
    }


    public double logStructurePrior2() {
        int[][] convertedModel = getConvertedCandiateModel();
//	    	System.out.println("****current candidate:");
//	    	System.out.println(printArray(convertedModel));
//	    	System.out.println(source_clean_BayesNet);
        double k = 1.0/(1+ adjustedPriorSampleSize);
        //	System.out.println("m_PriorEquivalentSampleSize "+m_PriorEquivalentSampleSize);
        int sizeParentDiff = 0;
        //Save names of nodes in source_clean_BayesNet
        ArrayList<String> nodesInSourceClean = new ArrayList<String>();
        //For source_clean_BayesNet, this table saves a node's name and its index
        HashMap<String,Integer> sourceNodesNameIndexTable = new HashMap<String,Integer>();
        Instances sourceCleanInstances = source_clean_BayesNet.m_Instances;
        for (int jNode=0; jNode<sourceCleanInstances.numAttributes(); jNode++){
            String jNodeName = sourceCleanInstances.attribute(jNode).name();
            nodesInSourceClean.add(jNodeName);
            sourceNodesNameIndexTable.put(jNodeName, jNode);
        }
        for (int iNode=1; iNode<=numberOfNodes; iNode++){
            String iNodeName = nodeInfo[iNode].name;
            //	System.out.println("iNode:"+iNode+" "+iNodeName);
            if (nodesInSourceClean.contains(iNodeName)){
                //		System.out.println("iNode:"+iNode+" "+iNodeName);
                List<String> parentSetInSource = new ArrayList<String>();
                int iNodeIndexInSource = sourceNodesNameIndexTable.get(iNodeName);
                //		System.out.println(iNodeIndexInSource+"");
                ParentSet temp = source_clean_BayesNet.getParentSet(iNodeIndexInSource);
                //		System.out.println("number of parents:"+temp.getNrOfParents());
                int[] parentsInSourceClean = temp.getParents();
                //		System.out.println("parents in source clean:");
                for (int i=0; i<temp.getNrOfParents(); i++){
                    int indexParentInSource = parentsInSourceClean[i];
                    //			System.out.println("parent:"+sourceCleanInstances.attribute(indexParentInSource).name());
                    parentSetInSource.add(sourceCleanInstances.attribute(indexParentInSource).name());
                }
                //		System.out.println("parents in candidate:");
                List<String> parentSetInCandiate = new ArrayList<String>();
                int[] parentsInCandidate = convertedModel[iNode];
                int numberOfParentInCandidate = parentsInCandidate[0];
                //		System.out.println("number of parents:"+numberOfParentInCandidate);
                for (int i=1; i<=numberOfParentInCandidate; i++){
                    int indexParentInCandidate = parentsInCandidate[i];
                    //			System.out.println(indexParentInCandidate+"");
                    String theParent = nodeInfo[indexParentInCandidate].name;
                    //			System.out.println("		" + theParent);
                    if (nodesInSourceClean.contains(theParent)){
                        parentSetInCandiate.add(theParent);
                        //				System.out.println("		add" + theParent);
                    }
                }
                //		System.out.println("parent set in source:");
                //		printList(parentSetInSource);
                //		System.out.println("parent set in candidate:");
                //		printList(parentSetInCandiate);
                sizeParentDiff  = sizeParentDiff + difference(parentSetInSource, parentSetInCandiate);
                // 		System.out.println("difference " + difference(parentSetInSource, parentSetInCandiate));
            }
        }
        //    	System.out.println("K:" + k);
        System.out.println("sizeParentDiff:" + sizeParentDiff);
        double temp2 = Math.log(k);
        double temp = sizeParentDiff * temp2;
        //System.out.println("structure score:" + temp + "temp2 " + temp2);
        return sizeParentDiff * Math.log(k) ;
    }
    /**
     * This method gets a converted candidate model by moving parents of the target node to the children,
     * and adding links between them.
     * This converted candidate model is used for structure scoring purpose later.
     * @return
     */
    private int[][] getConvertedCandiateModel(){
        int[][] convertedModel = copyParentArray();
        convertedModel = sortArray(convertedModel);
        // 	int[] parentOfTargetNode = convertedModel[targetNode];
        //if the targetNode has parents, then move them to children
        int lastParent=-1;
        int currentParent=-1;
        if (convertedModel[targetNode][0]>0){
            //from last parent
            for (int i = convertedModel[targetNode][0]; i>=1; i--){
                currentParent = convertedModel[targetNode][i];
                //remove this parent
                convertedModel[targetNode][i]=0;
                convertedModel[targetNode][0]--;
                //add target as its parent
                convertedModel[currentParent][0]++;
                convertedModel[currentParent][convertedModel[currentParent][0]] = targetNode;
                //add a link from currentParent to lastParent
                if (lastParent>0){
                    convertedModel[lastParent][0]++;
                    convertedModel[lastParent][convertedModel[lastParent][0]] = currentParent;
                }
                lastParent = currentParent;
            }
        }
        return convertedModel;
    }

    private int[][] copyParentArray(){
        int[][] copy = new int[numberOfNodes + 1][maxParents + 1];
        for (int i=0; i<parents.length; i++){
            for (int j=0; j<parents[i].length; j++){
                copy[i][j]=parents[i][j];
            }
        }
        return copy;
    }

    public int difference(List<String> listA, List<String> listB){
        List<String> union = Stream.concat(listA.stream(), listB.stream()).distinct().sorted().collect(Collectors.toList());
        List<String> intersection = listA.stream().filter(listB::contains).collect(Collectors.toList());
        List<String> unionDiffintersection = union.stream().filter(i -> !intersection.contains(i)).collect(Collectors.toList());
        return unionDiffintersection.size();
    }

    private void printList(List<String> listA){
        String toPrint = "";
        if (listA.size()>0) {
            for (int i=0; i<listA.size()-1; i++){
                toPrint = toPrint + listA.get(i)+",";
            }
            toPrint = toPrint+","+listA.get(listA.size()-1);
        }
        System.out.println(toPrint);
    }




    /**
     * For iNode in oneNetInstances, get the corresponding index for anotherNetInstances
     * @param oneNetInstances
     * @param iNode
     * @param anotherNetInstances
     * @return
     */
    private static int getIndexNodeForAnotherInstances(Instances oneNetInstances, int iNode, Instances anotherNetInstances){
        String iNodeName = oneNetInstances.attribute(iNode).name();
        for (int index=0; index<anotherNetInstances.numAttributes(); index++){
            String name = anotherNetInstances.attribute(index).name();
            if (iNodeName.equals(name)){
                return index;
            }
        }
        return -1;
    }


    /**
     * Use parents[][] to update m_BayesNet and get targetDataInstances to pair with it.
     * @param targetDataInstances
     * @throws Exception
     */
    public void updateBayesNet(Instances targetDataInstances) throws Exception{
        //re-initiated a m_BayesNet that is comparable to the targetInstances
        m_BayesNet = new BayesNet();
        m_BayesNet.m_Instances = targetDataInstances;
        m_BayesNet.initStructure();
        for(int i = 1; i < parents.length-1;i++){
            //	System.out.println("A"+i+" ("+parents[i][0]+") : ");
            if(parents[i][0] > 0){
                for(int j = 1; j<=parents[i][0]; j++){
                    //System.out.print("A"+parents[i][j]+" ");
                    //old code did not check whether the parentset already contains the node
                    //m_BayesNet.getParentSet(i-1).addParent(parents[i][j]-1, m_BayesNet.m_Instances);
                    //Ye new code are below.
                    ParentSet currentSet = m_BayesNet.getParentSet(i-1);
                    int iNode = parents[i][j]-1; // Note: Kevin's code index = weka code index + 1, weka code index = kevin code index - 1
                    if (!currentSet.contains(iNode)){
                        currentSet.addParent(iNode, m_BayesNet.m_Instances);
                    }
                }
            }
            //System.out.println();
        }
    }


    /**
     * Ye method: use net to update parents[][], parentPresent[],
     * childPresent[], numberOfChildren, children[], map[]
     * Use the incorporateNodeScores(theChild) to get lnChildren[][] ready
     * Notice: nodes may have different index in source_clean_BayesNet and targetInstances
     */
    public void updateParents(BayesNet net){
        parents = new int[numberOfNodes + 1][maxParents + 1];
        //	int nNode = m_BayesNet.getNrOfNodes();
        int nNode = targetInstances.numAttributes();
        for(int iNode = 0; iNode <= nNode-1;iNode++){
            int iNodeInSourceCleanNet = getIndexNodeForAnotherInstances(targetInstances,iNode,net.m_Instances);
            //If the Node appear in source_clean_BayesNet. Get its parent set infor and map it back to parents[][]
            if (iNodeInSourceCleanNet>=0){
                ParentSet currentSet = net.getParentSet(iNodeInSourceCleanNet);
                int setSize = currentSet.getNrOfParents();
                if (setSize>0){
                    //	System.out.println("setSize" + setSize);
                    parents[iNode+1][0] = setSize;
                    Integer[] list = new Integer[setSize];
                    for (int k=0; k<list.length; k++){
                        int oneParentIndexInSourceCleanNet = currentSet.getParent(k);
                        int oneParentIndex = getIndexNodeForAnotherInstances(net.m_Instances,oneParentIndexInSourceCleanNet,targetInstances);
                        list[k]=oneParentIndex;
                    }
                    //	Arrays.sort(list, Collections.reverseOrder());
                    //	System.out.println("listSize" + list.length);
                    for (int j=0; j<=list.length-1; j++){
                        int theChild = iNode+1;
                        int theParent = list[j]+1;
                        parents[theChild][j+1] = theParent;
                        if ((!parentPresent[theParent]) && (theChild==targetNode)) {parentPresent[theParent]=true;}
                        if ((!childPresent[theChild]) && (theParent==targetNode)) {childPresent[theChild]=true;}
                        if (theParent==targetNode) {
                            numberOfChildren++;
                            children[numberOfChildren] = theChild;
                            map[theChild]=numberOfChildren;
                        }
                        //	System.out.println(parents[iNode+1][j+1]);
                    }
                }
                else if (setSize==0) {
                    parents[iNode+1][0]=0;
                }
            }
        }
        System.out.println(printParents());
        //Ye added to get lnChildrenProb[][] ready for scoring the initial model
        //do it again to incorporateChildren infor for calculating scoreTarget
        for(int iNode = 0; iNode <= nNode-1;iNode++){
            int theChild = iNode+1;
            int iNodeInSourceCleanNet = getIndexNodeForAnotherInstances(targetInstances,iNode,net.m_Instances);
            if (iNodeInSourceCleanNet>=0){
                ParentSet currentSet = net.getParentSet(iNodeInSourceCleanNet);
                int setSize = currentSet.getNrOfParents();
                if (setSize>0){
                    Integer[] list = new Integer[setSize];
                    for (int k=0; k<list.length; k++){
                        int oneParentIndexInSourceCleanNet = currentSet.getParent(k);
                        int oneParentIndex = getIndexNodeForAnotherInstances(net.m_Instances,oneParentIndexInSourceCleanNet,targetInstances);
                        list[k]=oneParentIndex;
                    }
                    for (int j=0; j<=list.length-1; j++){
                        int theParent = list[j]+1;
                        if (theParent==targetNode) {
                            incorporateNodeScores(theChild);
                        }
                    }
                }
            }
        }
    }



    /*Ye added method to print current int[][] parents*/
    private String printParents(){
        String model = "";
        int numberOfNode = m_BayesNet.getNrOfNodes();
        //Initialize BayesNet to single variable
        for(int iNode = 0; iNode < numberOfNode; iNode++) {
            model +=  m_BayesNet.getNodeName(iNode) + " ("+m_BayesNet.getCardinality(iNode)+")";
            //System.out.println(m_BayesNet.getNodeName(iNode) + " ("+m_BayesNet.getCardinality(iNode)+")");
            int iNodeEBMC = iNode+1;
            if(parents[iNodeEBMC][0] > 0){
                // System.out.println("parents[iNodeEBMC][0]: " + parents[iNodeEBMC][0]);
                model += " this node's parents: ";
                for(int jNode = 1; jNode <= parents[iNodeEBMC][0]; jNode++){
                    int pNodeEBMC = parents[iNodeEBMC][jNode];
                    //	System.out.println(pNodeEBMC);
                    int pNode = pNodeEBMC-1;
                    //	System.out.println("pNode: "+pNode);
                    model += m_BayesNet.getNodeName(pNode);
                    if((jNode+1) < parents[iNodeEBMC].length-1){
                        model += ", ";
                    }
                }
            }
            model += "\n";
        }
        return model;
    }

    /*Ye added method to print current int[][] parents*/
    private String printArray(int[][] table){
        String model = "";
        //Initialize BayesNet to single variable
        for(int iNode = 0; iNode < numberOfNodes; iNode++) {
            //System.out.println(m_BayesNet.getNodeName(iNode) + " ("+m_BayesNet.getCardinality(iNode)+")");
            int iNodeEBMC = iNode+1;
            model += nodeInfo[iNodeEBMC].name;
            if(table[iNodeEBMC][0] > 0){
                // System.out.println("parents[iNodeEBMC][0]: " + parents[iNodeEBMC][0]);
                model += " parents: ";
                for(int jNode = 1; jNode <= table[iNodeEBMC][0]; jNode++){
                    int pNodeEBMC = table[iNodeEBMC][jNode];
                    //	System.out.println(pNodeEBMC);
                    //	System.out.println("pNode: "+pNode);
                    model += nodeInfo[pNodeEBMC].name;
                    if((jNode+1) < table[iNodeEBMC].length-1){
                        model += ", ";
                    }
                }
            }
            model += "\n";
        }
        return model;
    }

    private int[][] sortArray(int[][] array) {
        int n = array[targetNode][0];

        if (n > 1) {
            int L = (n / 2) + 1;
            int ir = n;

            while (true) {
                int rra;

                if (L > 1) {
                    L--;
                    rra = array[targetNode][L];
                } else {

                    rra = array[targetNode][ir];
                    array[targetNode][ir] = array[targetNode][1];
                    ir--;

                    if (ir == 1) {
                        array[targetNode][1] = rra;
                        return array;
                    }
                }

                int i = L;
                int j = 2 * L;
                while (j <= ir) {
                    if (j < ir) {
                        if (array[targetNode][j] < array[targetNode][j + 1]) {
                            j++;
                        }
                    }
                    if (rra < array[targetNode][j]) {
                        array[targetNode][i] = array[targetNode][j];
                        i = j;
                        j = 2 * j;
                    } else {
                        j = ir + 1;
                    }
                }

                array[targetNode][i] = rra;
            }  // end of while
        }
        return array;
    }







    /**
     * Returns the number of children for a given node
     *
     * @param node
     * @return
     */
    protected int getNrOfChildren(int node){
        int numberOfChildren = 0;
//		for(int i = 0; i < m_BayesNet.getNrOfNodes(); i++){
//			if (m_BayesNet.getParentSet(i).contains(node)){
//				numberOfChildren++;
//			}
//		}
        for (int i = 1; i <= numberOfNodes; i++){
            for (int j=1; j<=parents[i][0]; j++){
                if (parents[i][j]==node){
                    numberOfChildren++;
                }
            }
        }
        return numberOfChildren;
    }





/*******************Kevin artuo method **************************/


    /**
     * Derives the probability distribution over the target node
     * given the values of its parents in casei. The cases from
     * 1 to casei-1 are used to parameterize this predictive distribution.
     *
     * @param casei
     * @param node
     * @param k
     * @param v
     */
    protected void deriveNodeProbs(int casei, int node, double k, double v) {
        int numberOfParents, parentValue;
        int ctPtr, cPtr, ptr;

        numberOfParents = parents[node][0];

        for (int i = 1; i <= numberOfParents; i++)
            //Ye note This array saves the values of node's parents in case i
            values[i] = cases[casei][parents[node][i]];
        ctPtr = 1;
        for (int i = 1; i <= numberOfParents; i++) {
            parentValue = values[i];
            ptr = countsTree[ctPtr + parentValue - 1]; //Ye note Here, it already puts the parent info in case i into consideration.

            if (ptr > 0)
                ctPtr = ptr;
            else {  // there are no previous cases that match the current parent values of node, so return a uniform distribution
                for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++)
                    nodeProb[nodeValue] = k / v;

                return;
            }
        }
        cPtr = ctPtr;

        double b = 0;
        for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++)
            b += counts[cPtr + nodeValue - 1];

        for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++) {
            double a = counts[cPtr + nodeValue - 1];
            nodeProb[nodeValue] = (a + k) / (b + v); // nodeProb[nodeValue] = P(node = nodeValue | parents_node, case1,...,casei-1)
        }
    }

    protected void addChild_EBMC1(int child) {
        if (!childPresent[child]) {
            numberOfChildren += 1;
            children[numberOfChildren] = child;
            map[child] = numberOfChildren;
            childPresent[child] = true;
        }
        //Ye note: look like, parentPresent is used to temporally indicate whether some node is a parent of the child in this method
        for (int i = 1; i <= parents[child][0]; i ++) {
            parentPresent[parents[child][i]] = true;
        }

        if (parents[child][0] == 0) {
            parents[child][0]++;
            parents[child][parents[child][0]] = targetNode;
        }

        for (int i = 1; i <= parents[targetNode][0]; i++) {
            int targetParent = parents[targetNode][i];

            if (!parentPresent[parents[targetNode][i]]) {
                // newChild's parents will be the nodes to its left among the parents of the target node.
                if (targetParent == child) {
                    break;
                }

                parents[child][0]++;
                parents[child][parents[child][0]] = targetParent;
            }
        }

        incorporateNodeScores(child);

        // reset parentPresent for its future use
        for (int i = 1; i <= parents[child][0]; i++) {
            parentPresent[parents[child][i]] = false;
        }
    }

    protected void initializeFileCase(int node) {
        if (parents[node][0] > 0) {
            int firstParentSize = nodeDimension[parents[node][1]];

            for (int i = 1; i <= firstParentSize; i++)
                countsTree[i] = 0;

            countsTreePtr = firstParentSize + 1;
            countsPtr = 1;
        } else {
            countsTreePtr = 1;
            countsPtr = nodeDimension[node] + 1;

            for (int i = 1; i <= nodeDimension[node]; i++)
                counts[i] = 0;
        }
    }

    /**
     * This creates a tree that stores and indexes cases.
     * The branches of the tree correspond values of the parent nodes.
     * The leaves of the tree correspond to counts of target given the parent values.
     *
     * @param node
     * @param casei
     */
    protected void fileCase(int node, int casei) {
        int parent = 0;
        int parentValue = 0;
        int cPtr = 0;
        int parenti = 0;
        int nodeValue = cases[casei][node];
        int numberOfParents = parents[node][0];
        int ctPtr = 1;
        int ptr = 1;

        for (int i = 1; i <= numberOfParents; i++) {
            parent = parents[node][i];
            parentValue = cases[casei][parent];
            ptr = countsTree[ctPtr + parentValue - 1];

            if (ptr > 0)
                ctPtr = ptr;
            else {
                parenti = i;
                break;
            }
        }

        if (ptr > 0) {
            cPtr = ctPtr;
            counts[cPtr + nodeValue - 1]++;
        } else {
            // GrowBranch
            for (int i = parenti; i <= numberOfParents; i++) {
                parent = parents[node][i];
                parentValue = cases[casei][parent];

                if (i == numberOfParents)
                    countsTree[ctPtr + parentValue - 1] = countsPtr;
                else {
                    countsTree[ctPtr + parentValue - 1] = countsTreePtr;

                    for (int j = countsTreePtr; j <= (countsTreePtr + nodeDimension[parents[node][i + 1]] - 1); j++)
                        countsTree[j] = 0;

                    ctPtr = countsTreePtr;
                    countsTreePtr += nodeDimension[parents[node][i + 1]];

                }
            }

            for (int j = countsPtr; j <= (countsPtr + nodeDimension[node] - 1); j++)
                counts[j] = 0;

            cPtr = countsPtr;

            countsPtr += nodeDimension[node];

            // end of GrowBranch
            counts[cPtr + nodeValue - 1]++;
        }  // end of else statement
    }//fileCase

    private class IntegerComparator implements Comparator<Integer> {
        public int compare(Integer o1, Integer o2) {
            return o2.compareTo(o1);
        }
    }


    /**
     * Extract data from a particular ARFF file.
     *
     */
    protected void readCases(){

        //numberOfNodes = m_BayesNet.getNrOfNodes(); //wrong

        numberOfNodes = targetInstances.numAttributes();
        nodeDimension = new byte[numberOfNodes + 1];
        int index = 1;

        for(int iNode = 0; iNode < numberOfNodes; iNode++){
            //nodeDimension[index++] = (byte) m_BayesNet.getCardinality(iNode);
            nodeDimension[index++] = (byte) targetInstances.attribute(iNode).numValues();
        }

        //	numberOfCases = m_BayesNet.m_Instances.numInstances();
        numberOfCases = targetInstances.numInstances();

        cases = new byte[numberOfCases + 1][numberOfNodes + 1];
        for (int row = 1; row <= numberOfCases; row++) {
            for(int col = 1; col <= numberOfNodes; col++){
                cases[row][col] = toByte(row-1, col-1);
                //	System.out.print(cases[row][col]+" ");
            }
            // System.out.println();
        }

        nameIndexTable = new HashMap<String,Integer>();
        for (int i=0; i<targetInstances.numAttributes(); i++){
            String variableName = targetInstances.attribute(i).name();
            nameIndexTable.put(variableName, new Integer(i));
        }

    }


    protected void initVariables(){
        readNodeInfo(numberOfNodes); //initiate NodeInfo array

        maxParents = getMaxNrOfParents();
        maxChildren = getMaxNrOfChildren();

        maxValue = getMaxValue();
        maxCell = maxParents * maxValue * numberOfCases;
        map = new int[numberOfNodes + 1];
        parents = new int[numberOfNodes + 1][maxParents + 1];
        values = new int[numberOfNodes + 1];

        newChildren = new int[MAXNEWCHILDREN + 1];

        children = new int[maxChildren + 1];
        nodeProb = new double[maxValue + 1];
        targetCounts = new int[maxValue + 1];

        childPresent = new boolean[numberOfNodes + 1];
        parentPresent = new boolean[numberOfNodes + 1];
        for (int i = 1; i <= numberOfNodes; i++) {
            childPresent[i] = false;
            parentPresent[i] = false;
            parents[i][0] = 0;
        }

        fileCaseCache = new FileCaseRecord[maxChildren + 1];
        for (int i = 1; i <= maxChildren; i++)
            fileCaseCache[i] = new FileCaseRecord(maxCell);

        counts = new int[maxCell + 1];
        countsTree = new int[maxCell + 1];
        lnChildProb = new double[numberOfCases + 1][maxValue + 1][maxChildren + 1];

        numberOfModelsScored = 0;
        numberOfChildren = 0;

        childPresent = new boolean[numberOfNodes + 1];
        newChildPresent = new boolean[numberOfNodes + 1];
        parents = new int[numberOfNodes + 1][maxParents + 1];
        for (int i = 1; i <= numberOfNodes; i++) {
            childPresent[i] = false;
            newChildPresent[i] = false;
            parents[i][0] = 0;
        }

        firstCase = lowerBound = 1;
        lastCase = upperBound = numberOfCases;

        targetNode = numberOfNodes;

        lnChildrenProb = new double[numberOfCases + 1][maxValue + 1];
        for (int casei = 1; casei <= lastCase; casei++)
            for (int j = 1; j <= nodeDimension[targetNode]; j++)
                lnChildrenProb[casei][j] = 0;
    }


    /**
     * Searches for an additional, new set of children that improve the prediction of the target node
     * in light of the existing children that have already been "locked in".
     *
     * @param targetNode
     * @return
     */
    protected boolean searchChildren(int targetNode) {
        boolean allTrue, improvement;
        double bestScore, score;
        int bestChild;

        parents[targetNode][0] = 0;
        //	System.out.println("prior: "+prior());
        nodeScore = scoreNode(targetNode) + prior_EBMC2();  // Score the targetnode with current children (if any)
        //if(m_BayesNet.getDebug() == true){
        System.out.println("score: "+nodeScore);
        //}
        numberOfModelsScored++;

        newChildren[0] = 0;  // newChildren is a list of new children being added to the model.

        // It can be considered as a kind of generalized rule that predicts the target.
        improvement = false;
        //if(m_BayesNet.getDebug() == true){
        System.out.println("starting search for an additional predictor");
        //}
        while (newChildren[0] < MAXNEWCHILDREN) {
            bestScore = 1;
            bestChild = 0;
            allTrue = true;

            for (int child = 1; child <= numberOfNodes; child++) {
                if (child != targetNode && (!childPresent[child] || (childPresent[child] && !newChildPresent[child] && newChildren[0] > 0))) {
                    allTrue = false;
                    //System.out.println("numberOfChildren: "+numberOfChildren);

                    if(numberOfChildren < maxChildren){ //added by Arturo!!
                        addChild_EBMC2(child);  // adds child as a child in the newChildren "rule" being constructed and tested
                    }
                    score = tallyModelScore();  // scores the new "rule" consisting of the newChildren
                    numberOfModelsScored++;

                    if (score > bestScore || bestScore == 1) {
                        bestChild = child;
                        bestScore = score;
                    }

                    removeChild(child);  // removes child as a child of the newChildren "rule"
                }
            }

            // break out of While, because no children can be added to the rule
            if (allTrue)
                break;

            if (bestScore > nodeScore) {
                nodeScore = bestScore;
                addChild_EBMC2(bestChild);
                improvement = true;

                //if(m_BayesNet.getDebug() == true){
                System.out.print("   current new predictors: ");

                for (int j = 1; j <= newChildren[0]; j++)
                    System.out.print(nodeInfo[newChildren[j]].name + " ");
                System.out.println("   score:"+nodeScore);
                //}

            } else
                break;
        }  // end of while

        if (improvement && (numberOfChildren < maxChildren))
            return true;
        else
            return false;
    } //searchChildren


    protected void addChild_EBMC2(int child) {
        newChildPresent[child] = true;

        // When "child" is already part of the permanent model, we make it a parent
        // of all the nodes in the current newChildren rule. By doing so, we avoid
        // having to add new parents to existing nodes in the permanent model, which
        // might significantly disrupt the model score in a bad way.
        if (childPresent[child]) {
            childJustAddedFlag = false;

            for (int i = 1; i <= newChildren[0]; i++) {
                int existingNewChild = newChildren[i];

                if (parents[existingNewChild][0] < maxParents) {
                    removeNodeScores(existingNewChild);
                    parents[existingNewChild][0]++;
                    parents[existingNewChild][parents[existingNewChild][0]] = child;
                    incorporateNodeScores(existingNewChild);
                }
            }
        } else {  // when child is not part of the permanent model, we make it a child of each the nodes in the current newChildren rule
            childJustAddedFlag = true;
            numberOfChildren++;
            //System.out.println("numberOfChildren = "+numberOfChildren);
            children[numberOfChildren] = child;
            map[child] = numberOfChildren; // Ye comment: the order of the child. for example X4 is the 2nd child of the target map[4]=2
            childPresent[child] = true;
            parents[child][0] = 1;
            parents[child][parents[child][0]] = targetNode;

            int min = Math.min(newChildren[0], maxParents - 1);
            for (int i = 1; i <= min; i++) {
                int existingNewChild = newChildren[i];
                parents[child][0]++;
                parents[child][parents[child][0]] = existingNewChild;
            }

            newChildren[0]++;
            newChildren[newChildren[0]] = child;
            incorporateNodeScores(child);
        }
    }




    /**
     * Adds a given parent to a node
     *
     * @param node
     * @return void

    protected void addParent(int node, int parent){
    if(!isParentXofY(parent, node) && node != parent){
    m_BayesNet.getParentSet(node).addParent(parent, inst);
    }
    }*/

    /**
     * Calc Node Score With AddedParent
     *
     * @param nNode node for which the score is calculate
     * @param nCandidateParent candidate parent to add to the existing parent set
     * @return log score

    protected double calcScoreWithExtraParent(int nNode, int nCandidateParent) {
    ParentSet oParentSet = m_BayesNet.getParentSet(nNode);

    // sanity check: nCandidateParent should not be in parent set already
    if (oParentSet.contains(nCandidateParent)) {
    return -1e100;
    }

    // set up candidate parent
    oParentSet.addParent(nCandidateParent, m_BayesNet.m_Instances);

    // calculate the score
    double logScore = scoreNode(targetNode) + prior(targetNode);

    // delete temporarily added parent
    oParentSet.deleteLastParent(m_BayesNet.m_Instances);

    return logScore;
    } // CalcScoreWithExtraParent
     */


    /**
     * Calc Node Score With Parent Deleted
     *
     * @param nNode node for which the score is calculate
     * @param nCandidateParent candidate parent to delete from the existing parent set
     * @return log score

    protected double calcScoreWithMissingParent(int nNode, int nCandidateParent) {
    ParentSet oParentSet = m_BayesNet.getParentSet(nNode);

    // sanity check: nCandidateParent should be in parent set already
    if (!oParentSet.contains( nCandidateParent)) {
    return -1e100;
    }

    // set up candidate parent
    int iParent = oParentSet.deleteParent(nCandidateParent, m_BayesNet.m_Instances);

    // calculate the score
    double logScore = scoreNode(targetNode) + prior(targetNode);;

    // restore temporarily deleted parent
    oParentSet.addParent(nCandidateParent, iParent, m_BayesNet.m_Instances);

    return logScore;
    } // CalcScoreWithMissingParent
     */

    /**
     * Remove the parents of a node, keeps children
     *
     * @param node
     * @return

    public void deleteAllParents(int node){

    int parents[] = m_BayesNet.getParentSet(node).getParents();

    for(int i=0; i < parents.length; i++){
    m_BayesNet.getParentSet(node).deleteParent(parents[i], inst);
    }
    }*/

    /**
     * Removes a given parent from a node
     *
     * @param node
     * @return

    protected void deleteParent(int node, int parent){
    if(isParentXofY(parent, node) && node != parent){
    m_BayesNet.getParentSet(node).deleteParent(parent, inst);
    }
    }*/



    /**
     * @return a string to describe the expectedParentsOfTarget option.
     */
    public String expectedParentsOfTargetTipText() {
        return "Set the number of parents that the target node is expected to have.";
    } // expectedParentsOfTarget

    /**
     * Returns the children for a given node
     *
     * @param node
     * @return

    protected int[] getChildren(int node){
    int[] children = new int[getNrOfChildren(node)];

    int index = 0;

    for(int i = 0; i < m_BayesNet.getNrOfNodes(); i++){
    if (m_BayesNet.getParentSet(i).contains(node)){
    children[index] = i;
    index++;
    }
    }

    return children;
    }*/

    /**
     * Gets the Expected Parents of Target
     *
     * @return the Exp
     */
    public int getExpectedParentsOfTarget() {
        return m_ExpectedParentsOfTarget;
    }



    /**
     * Gets whether to init as naive bayes
     *
     * @return whether to init as naive bayes

    public boolean getInitAsNaiveBayes() {
    return m_bInitAsNaiveBayes;
    }*/



    /**
     * Gets the Maximum number of Children
     *
     * @return the maxChildren
     */
    public int getMaxNrOfChildren() {
        return m_MaxNrOfChildren;
    }



    /**
     * Gets the Maximum number of parents
     *
     * @return the maxParents
     */
    public int getMaxNrOfParents() {
        return m_MaxNrOfParents;
    }

    protected byte getMaxValue() {
        byte max = 0;

        for (byte value : nodeDimension)
            if (max < value)
                max = value;

        return max;
    }


    /**
     * Returns the number of parents for a given node
     *
     * @param node
     * @return
     */
//	protected int getNrOfParents(int node){
//		return m_BayesNet.getParentSet(node).getNrOfParents();
//	}

    /**
     * Gets the current settings of the search algorithm.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String [] getOptions() {
        Vector<String> result  = new Vector<String>();

        result.add("-T"); // expected parents of target
        result.add("" + getExpectedParentsOfTarget());

        result.add("-P"); // Maximum number of parents
        result.add("" + getMaxNrOfParents());

        result.add("-C"); // Maximum number of children
        result.add("" + getMaxNrOfChildren());

        result.add("-S"); // scoring metric

        switch (m_ScoreMetric) {
            case(0):
                result.add("K2");
                break;
            case(1):
                result.add("BDeu");
            case(2):
                result.add("priorModelApproach");
                result.add("-E"); // prior equivalent sample size
                //result.add("" + getAdjustedPriorSampleSize());
                break;
        }

        return (String[]) result.toArray(new String[result.size()]);
    }

    /**
     * Returns the i-th parent of a node
     *
     * @param node
     * @return

    protected int getParent(int node, int index){
    return m_BayesNet.getParentSet(node).getParent(index);
    }*/


    /**
     * Returns the parents for a given node
     *
     * @param node
     * @return

    protected int[] getParents(int node){
    return m_BayesNet.getParentSet(node).getParents();
    }*/

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 8034 $");
    }

    /**
     * get quality measure to be used in searching for networks.
     * @return quality measure
     */
    public SelectedTag getScoreMetric() {
        return new SelectedTag(m_ScoreMetric, SCORING_METRICS);
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "G.F. Cooper and P. Hennings-Yeomans and "
                + "S. Visweswaran and M. Barmada");
        result.setValue(Field.TITLE, "An efficient bayesian method for predicting "
                + "clinical outcomes from genome-wide data");
        result.setValue(Field.YEAR, "2010");
        result.setValue(Field.PAGES, "127-131");
        result.setValue(Field.PUBLISHER, "AMIA Anual Symposium Proceedings");

        return result;
    }

    /**
     * This will return a string describing the search algorithm.
     * @return  a description of the data generator suitable for displaying in
     *         the explorer/experimenter gui
     */
    public String globalInfo() {

        return "This algorithm performs greedy search in a subspace of Bayesian "
                + "Networks to find the one that best predicts a target node.\n\n"
                + "For more information refer to:\n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Check if a given node has children
     *
     * @param node
     * @return

    protected boolean hasChildren(BayesNet bn, int node){
    boolean children = false;
    int i=0;
    while( children == false & (i < bn.getNrOfNodes()) ){
    if (bn.getParentSet(i).contains(node)){
    children = true;
    }
    i++;
    }
    return children;
    }*/

    /**
     * Check if a given node has parents
     *
     * @param node
     * @param bn the model
     * @return boolean state if the node has parents

    protected boolean hasParents(BayesNet bn, int node){
    boolean parents = false;
    if (bn.getParentSet(node).getNrOfParents() > 0){
    parents = true;
    }
    return parents;
    }*/



    /**
     * @return a string to describe the InitAsNaiveBayes option.
     */
    public String initAsNaiveBayesTipText() {
        return "Not used in this method, set as False.";
    }




    /**
     * Check if a given node has parents
     *
     * @param node
     * @return

    protected boolean isChildrenXofY(int xNode, int yNode){
    boolean isChildren = false;
    if (m_BayesNet.getParentSet(xNode).contains(yNode)){
    isChildren = true;
    }
    return isChildren;
    }*/

    /**
     * Check if a given node has parents
     *
     * @param node
     * @return

    protected boolean isParentXofY(int xNode, int yNode){
    boolean isParent = false;
    if(m_BayesNet.getParentSet(yNode).contains(xNode)){
    isParent = true;
    }
    return isParent;
    }*/


    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>(0);

        newVector.addElement(new Option("\tExpected parents of target", "T", 2,
                "-P <nr of parents>"));
        newVector.addElement(new Option("\tMaximum number of parents", "P", 2,
                "-P <nr of parents>"));
        newVector.addElement(new Option("\tMaximum number of children", "C", 2,
                "-C <nr of children>"));
        newVector.addElement(new Option("\tPrior Equivalent Sample Size", "E", 0,
                "-E <pess>"));
        newVector.addElement(new Option("\tScoring Metric", "S", 0,
                "-S <scoring metric>"));

        Enumeration<Option> enu = listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        return newVector.elements();

    }

    /**
     * Takes ln(x) and ln(y) as input, and returns ln(x + y)
     *
     * @param lnX is natural log of x
     * @param lnY is natural log of y
     * @return natural log of x plus y
     */
    protected double lnXpluslnY(double lnX, double lnY) {
        double lnYminusLnX, temp;

        if (lnY > lnX) {
            temp = lnX;
            lnX = lnY;
            lnY = temp;
        }

        lnYminusLnX = lnY - lnX;

        //if (lnYminusLnX < MININUM_EXPONENT)
        if (lnYminusLnX < Double.MIN_EXPONENT) //Suggested change by Kevin
            return lnX;
        else
            return Math.log1p(Math.exp(lnYminusLnX)) + lnX;
    }


    /**
     * @return a string to describe the MarkovBlanketClassifier option.
     */
    public String markovBlanketClassifierTipText() {
        return "Not used in this method, set as False.";
    }



    /**
     * @return a string to describe the MaxNrOfChildren option.
     */
    public String maxNrOfChildrenTipText() {
        return "Set the maximum number of children that each node can have.";
    } // maxNrOfChildrenTipText

    /**
     * @return a string to describe the MaxNrOfParents option.
     */
    public String maxNrOfParentsTipText() {
        return "Set the maximum number of parents that each node can have.";
    } // maxNrOfParentsTipText


    /**
     * @return a string to describe the MaxNrOfParents option.
     */
    public String priorEquivalentSampleSizeTipText() {
        return "Ajust prior sample size is for prior model approach."
                + " Prior equivalent sample size used in the BDeu scoring measure only."
                + " This value is not needed in the K2 scoring.";
    } // maxNrOfParentsTipText



    protected double numberOfJointStates(int node) {
        double x = (double) nodeDimension[node];

        for (int i = 1; i <= parents[node][0]; i++)
            x *= (double) nodeDimension[parents[node][i]];

        return x;
    }



	/*
	 * protected String print(BayesNet bn){
	 *
		String model = "";
		//Initialize BayesNet to single variable
		for(int iNode = 0; iNode < bn.getNrOfNodes(); iNode++) {
			if(hasParents(bn, iNode) || hasChildren(bn, iNode) || iNode == targetNode){
				model += bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")";
				if(bn.getParentSet(iNode).getNrOfParents() > 0){
					model += " --> ";
				}
				for(int jNode = 0; jNode < bn.getParentSet(iNode).getNrOfParents(); jNode++){
					int pNode = bn.getParentSet(iNode).getParent(jNode);

					model += bn.getNodeName(pNode);
					if((jNode+1) < bn.getParentSet(iNode).getNrOfParents()){
						model += ", ";
					}
				}

				model += "\n";
			}
			else{
				model += bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")\n";
			}
		}
		return model;
		//System.out.println("Score: " + scoreNode(targetNode));
		//System.out.println("----");
	}*/


	/*protected void printFull(BayesNet bn){
		//Initialize BayesNet to single variable
		for(int iNode = 0; iNode < bn.getNrOfNodes(); iNode++) {
			System.out.print("Node["+iNode+"]: "+bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")");
			if(bn.getParentSet(iNode).getNrOfParents() > 0){
				System.out.print(" --> ");
			}
			for(int jNode = 0; jNode < bn.getParentSet(iNode).getNrOfParents(); jNode++){
				int pNode = bn.getParentSet(iNode).getParent(jNode);

				System.out.print(bn.getNodeName(pNode));
				if((jNode+1) < bn.getParentSet(iNode).getNrOfParents()){
					System.out.print(", ");
				}
			}

			System.out.println("");
		}

		System.out.println("----");
	}*/

    //These prior methods need to be checked
    protected double prior_EBMC2() {
        double nc = (double) numberOfChildren;
        double nn = (double) (numberOfNodes - 1);
        double p = ((double) getExpectedParentsOfTarget()) / nn;  // nn equals the number of potential predictors of the target
        return (nc * Math.log(p) + (nn - nc) * Math.log(1 - p));
    }

//    public double prior(int node) {
//        double np = (double) parents[node][0];
//        double nc = (double) numberOfChildren;
//        double nn = (double) (numberOfNodes - 1);
//        double p = ((double) m_ExpectedParentsOfTarget) / nn;  // modify m_ExpectedParentsOfTarget part
//        // nn equals the number of potential predictors of the target
//        return ((np + nc) * Math.log(p) + (nn - nc - np) * Math.log(1 - p));
//    }

    protected double prior_EBMC1(int node) {
//		double np = (double) m_BayesNet.getNrOfParents(node);
//		double nc = (double) getNrOfChildren(node);
//		double nn = (double) (m_BayesNet.getNrOfNodes() - 1);
//		double p = ((double) m_ExpectedParentsOfTarget) / nn;  // nn equals the number of potential predictors of the target
        double nc = (double) getNrOfChildren(node);
        double nn = (double) (numberOfNodes - 1);
        double np = (double) parents[node][0];
        double p = ((double) m_ExpectedParentsOfTarget) / nn;  // nn equals the number of potential predictors of the target
        return ((np + nc) * Math.log(p) + (nn - nc - np) * Math.log(1 - p));
    }

    protected void pruneParents(int node) {
        System.out.println("check node " + nodeInfo[node].name);
        double modelScore = tallyModelScore();
        System.out.println("modelScore: "+ modelScore);
        double bestScore = 1;
        double score = 0;
        while (parents[node][0] > 0) {
            int numberOfParents = parents[node][0];
            int besti = 0;

            // remove i as a parent of node
            for (int i = 1; i <= numberOfParents; i++) {

                int parent = parents[node][i];
                System.out.println("if removing " + nodeInfo[parent].name + " as parent of " + nodeInfo[node].name);
                for (int j = (i + 1); j <= numberOfParents; j++)
                    parents[node][j - 1] = parents[node][j];
                parents[node][0]--;
                score = scoreNode2(node);
                System.out.println("modelScore: "+ score);
                if ((score > bestScore) || (bestScore == 1)) {
                    besti = i;
                    bestScore = score;
                }
                // add i back as a parent of node
                for (int j = numberOfParents; j >= (i + 1); j--)
                    parents[node][j] = parents[node][j - 1];
                parents[node][i] = parent;
                parents[node][0]++;
            }
            if (bestScore > modelScore) {
                //if(m_BayesNet.getDebug() == true){
                System.out.println("Pruning node "+nodeInfo[parents[node][besti]].name+" away from node "+nodeInfo[node].name);
                //}
                // remove the parent that in doing so most increases the score
                for (int j = (besti + 1 ); j <= numberOfParents; j++)
                    parents[node][j - 1] = parents[node][j];
                parents[node][0]--;
                modelScore = bestScore;
            } else
                break;
        }  // end of while-loop
        score = scoreNode2(node);
    }



    protected void readNodeInfo(int numOfNodes){
        nodeInfo = new NodeInfoRecord[numberOfNodes + 1];
        for (int i = 1; i <= numberOfNodes; i++) {
            //wrong: String name = 	m_BayesNet.getNodeName(i-1);
            //wrong:	int numberOfValues = m_BayesNet.getCardinality(i-1);
            String name = targetInstances.attribute(i-1).name();
            int numberOfValues = targetInstances.attribute(i-1).numValues();
            nodeInfo[i] = new NodeInfoRecord(name, numberOfValues);
            nodeInfo[i].value  = new String[numberOfValues+1];
            for (int k=1; k<=numberOfValues; k++){
                nodeInfo[i].value[k] = targetInstances.attribute(i-1).value(k-1);
                //System.out.println(name+","+nodeInfo[i].value[k]);
            }
        }

    }




    protected void removeChild(int child) {
        newChildPresent[child] = false;

        if (childJustAddedFlag) {
            numberOfChildren--;
            childPresent[child] = false;
            parents[child][0] = 0;
            newChildren[0]--;
            removeNodeScores(child);
            map[child] = 0;
        } else {
            for (int i = 1; i <= newChildren[0]; i++) {
                int existingNewChild = newChildren[i];
                removeNodeScores(existingNewChild);
                parents[existingNewChild][0]--;
                incorporateNodeScores(existingNewChild);
            }
        }
    }

    protected void removeNodeScores(int node) {
        int mapNode = map[node];

        for (int casei = firstCase; casei <= lastCase; casei++) {
            for (int targetValue = 1; targetValue <= nodeDimension[targetNode]; targetValue++) {
                double lnProb = lnChildProb[casei][targetValue][mapNode];
                lnChildrenProb[casei][targetValue] -= lnProb;
            }
        }
    }



    /**
     * Derives overall model score, using in the model "node" and its parents.
     * It differs from ScoreNode in its use in scoring only models with children of the target,
     * whereas ScoreNode scores parents of the target.
     *
     * @param node
     * @return
     */
    protected double scoreNode2(int node) {
        removeNodeScores(node);
        incorporateNodeScores(node);
        return tallyModelScore();
    }


    /**
     * Performs a greedy, forward stepping search for the highest scoring model,
     * according to the score that is returned by ScoreNode.
     *
     * @param node is the target node

    protected void searchParents(int node) {
    boolean allTrue;
    double bestScore, score, nodeScore;
    int bestParent, parent;

    System.out.println("Score with no parents:");
    nodeScore = scoreNode(node) + prior(node);  // Score the node with no parents.
    System.out.println("score: "+nodeScore+"\n----");

    while (getNrOfParents(node) < getMaxNrOfParents()) {
    bestScore = 1;
    bestParent = 0;
    allTrue = true;

    // The last conjunct above insures that the first parent of the target will not already be a child of the target.
    // This condition guarantees that the predictors (parents) of the target will be a unique set.
    // Otherwise, the search could get stuck in an infinite loop.
    for(int i = 0; i < inst.numAttributes(); i++) {
    if ((i != node) && !isParentXofY(i, node) && !isChildrenXofY(i, node)) {
    allTrue = false;
    parent = i;
    score = calcScoreWithExtraParent(node, parent);

    if ((score > bestScore) || (bestScore == 1)) {
    bestParent = parent;
    bestScore = score;
    }
    }
    }

    if (allTrue)
    break;

    if (bestScore > nodeScore) {
    nodeScore = bestScore;
    addParent(node, bestParent);
    } else
    break;
    }
    }
     */

    /**
     * Sets the Expected Parents of Target
     *
     * @param exp the expected parents of target
     */
    public void setExpectedParentsOfTarget(int exp) {
        this.m_ExpectedParentsOfTarget = exp;
    }


    /**
     * Sets the Maximum number of children
     *
     * @param setMaxChildren the children number of parents that any node is allowed to have
     */
    public void setMaxNrOfChildren(int maxChildren) {
        this.m_MaxNrOfChildren = maxChildren;
    }


    /**
     * Sets the Maximum number of parents
     *
     * @param setMaxParents the maximum number of parents that any node is allowed to have
     */
    public void setMaxNrOfParents(int maxParents) {
        this.m_MaxNrOfParents = maxParents;
    }

    /**
     * Parses a given list of options. <p/>
     *
     <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -T &lt;expected parents of target&gt;
     *  Maximum number of parents</pre>
     *
     * * <pre> -P &lt;nr of parents&gt;
     *  Maximum number of parents</pre>
     *
     * <pre> -C &lt;nr of children&gt;
     *  Maximum number of parents</pre>
     *
     * <pre> -mbc
     *  Applies a Markov Blanket correction to the network structure,
     *  after a network structure is learned. This ensures that all
     *  nodes in the network are part of the Markov blanket of the
     *  classifier node.</pre>
     *
     * <pre> -S [K2|BDeu]
     *  Score type (BAYES, BDeu)</pre>
     *
     *  * <pre> -E &lt;prior equivalent sample size;
     *  Pess</pre>
     *
     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        //m_bInitAsNaiveBayes = !(Utils.getFlag('N', options));

        setExpectedParentsOfTarget(Integer.parseInt(Utils.getOption('T', options)));

        setMaxNrOfParents(Integer.parseInt(Utils.getOption('P', options)));

        setMaxNrOfChildren(Integer.parseInt(Utils.getOption('C', options)));

        String sScore = Utils.getOption('S', options);

        if (sScore.compareTo("K2") == 0) {
            setScoreMetric(new SelectedTag(0, SCORING_METRICS));
        }
        if (sScore.compareTo("BDeu") == 0) {
            setScoreMetric(new SelectedTag(1, SCORING_METRICS));
            //setPriorEquivalentSampleSize(Integer.parseInt(Utils.getOption('E', options)));
        }



        //setOptions(options);
    }



    /**
     * set quality measure to be used in searching for networks.
     *
     * @param newScoreMetric the new score type
     */
    public void setScoreMetric(SelectedTag newScoreMetric) {
        if (newScoreMetric.getTags() == SCORING_METRICS) {
            m_ScoreMetric = newScoreMetric.getSelectedTag().getID();
        }
    }


    protected double tallyModelScore() {
        parents[targetNode][0] = 0;
        if (m_ScoreMetric==2 ) {
            return (scoreNode(targetNode) + logStructurePrior());
        }
        else return scoreNode(targetNode) + prior_EBMC1(targetNode);
        //if EBMC 2 return scoreNode(targetNode) + prior_EBMC2();
    }

    protected byte toByte(int instance, int attribute){
        byte bit = 0;
        String value = targetInstances.instance(instance).stringValue(attribute);
        bit = (byte) (targetInstances.attribute(attribute).indexOfValue(value) + 1);
        //	System.out.println(value+":"+bit);
        return bit;
    }

    /**
     * Removes arcs among children (of the target) that contribute negatively to the overall model score.
     * Greedily removes arcs of each child, one child at a time. Thus, it is quite myopic, yet fairly efficient.
     */
    protected void weedoutWeakArcs() {
        parents[targetNode][0] = 0;  // At this point, the model consists only of the target, children of the target, and selected arcs among the children.
        for (int i = 1; i <= numberOfChildren; i++)
            pruneParents(children[i]);
    }

    protected String print(BayesNet bn){

        String model = "";
        //Initialize BayesNet to single variable
        for(int iNode = 0; iNode < bn.getNrOfNodes(); iNode++) {
            if(hasParents(bn, iNode) || hasChildren(bn, iNode) || iNode == (bn.getNrOfNodes()-1)){
                model += bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")";
                if(bn.getParentSet(iNode).getNrOfParents() > 0){
                    model += " --> ";
                }
                for(int jNode = 0; jNode < bn.getParentSet(iNode).getNrOfParents(); jNode++){
                    int pNode = bn.getParentSet(iNode).getParent(jNode);

                    model += bn.getNodeName(pNode);
                    if((jNode+1) < bn.getParentSet(iNode).getNrOfParents()){
                        model += ", ";
                    }
                }

                model += "\n";
            }
            else{
                model += bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")\n";
            }
        }
        return model;
        //System.out.println("Score: " + scoreNode(targetNode));
        //System.out.println("----");
    }

    /**
     * Check if a given node has children
     *
     * @param node
     * @return
     */
    protected static boolean hasChildren(BayesNet bn, int node){
        boolean children = false;
        int i=0;
        while( children == false & (i < bn.getNrOfNodes()) ){
            if (bn.getParentSet(i).contains(node)){
                children = true;
            }
            i++;
        }
        return children;
    }

    /**
     * Check if a given node has parents
     *
     * @param node
     * @param bn the model
     * @return boolean state if the node has parents
     */
    protected static boolean hasParents(BayesNet bn, int node){
        boolean parents = false;
        if (bn.getParentSet(node).getNrOfParents() > 0){
            parents = true;
        }
        return parents;
    }

    /**.
     * This method is getting ajusted prior sample size for the prior model approach
     * @return
     */
//	public int getAdjustedPriorSampleSize() {
//			return m_SourceSampleSize;
//	}


//	public void setPriorEquivalentSampleSize(int m_PriorEquivalentSampleSize) {
//		this.adjustedPriorSampleSize = m_PriorEquivalentSampleSize;
//	}



    // Below is EBMC 1 portion
    /**
     * Performs a greedy, forward stepping search for the highest scoring model,
     * according to the score that is returned by ScoreNode.
     *
     * @param node is the target node
     */







    protected double timeMinutes() {
        return TimeUtility.getDurationInMinutes(startTime, new DateTime(System.currentTimeMillis()));
    }

    /**
     * Moves the parents of the target node to be children of the target node
     * and adds appropriate links among the children.
     * There may already be other children as well. If so, the new children
     * are integrated appropriately with the existing children.
     *
     * @return true
     */
    protected boolean moveParentsToChildren() {
        //  sortTargetParents(); //Ye note: descending sort parents. for example node 100 will be the first parent. node 2 will be the last.
        boolean structureGrew = false;
        for (int i = 1; i <= parents[targetNode][0]; i++) {
            int node = parents[targetNode][i];
            if (childPresent[node]) {
                if (!redundant(node)) {
                    updateChild(node);
                    structureGrew = true;
                }
            } else {
                addChild_EBMC1(node);
                structureGrew = true;
            }
        }
        if (structureGrew) {
            System.out.println("the new predictors have been incorporated into the model"); //a set of features has been added
        }
        //Below is Ye added.
        for (int i=0; i<parents[targetNode].length; i++){
            parents[targetNode][i]=0;
        }
        return structureGrew;
    }


    /**
     * descending sort parents. for example node 100 will be the first parent. node 2 will be the last.
     */
    protected void sortTargetParents() {
        int n = parents[targetNode][0];
        if (n > 1) {
            int L = (n / 2) + 1;
            int ir = n;
            while (true) {
                int rra;
                if (L > 1) {
                    L--;
                    rra = parents[targetNode][L];
                } else {
                    rra = parents[targetNode][ir];
                    parents[targetNode][ir] = parents[targetNode][1];
                    ir--;
                    if (ir == 1) {
                        parents[targetNode][1] = rra;
                        return;
                    }
                }
                int i = L;
                int j = 2 * L;
                while (j <= ir) {
                    if (j < ir) {
                        if (parents[targetNode][j] < parents[targetNode][j + 1]) {
                            j++;
                        }
                    }
                    if (rra < parents[targetNode][j]) {
                        parents[targetNode][i] = parents[targetNode][j];
                        i = j;
                        j = 2 * j;
                    } else {
                        j = ir + 1;
                    }
                }
                parents[targetNode][i] = rra;
            }  // end of while
        }
    }
    protected boolean redundant(int newChild) {
        boolean allSame = true;

        for (int i = 1; i <= parents[newChild][0]; i++) {
            int parent = parents[newChild][i];
            parentPresent[parent] = true;
        }

        for (int i = 1; i <= parents[targetNode][0]; i++) {
            int parent = parents[targetNode][i];

            if (parent == newChild) {
                break;
            }

            if (!parentPresent[parent]) {
                allSame = false;
                break;
            }
        }

        for (int i = 1; i <= parents[newChild][0]; i++) {
            int parent = parents[newChild][i];
            parentPresent[parent] = false;
        }

        return allSame;
    }
    protected void updateChild(int node) {
        removeNodeScores(node);
        addChild_EBMC1(node);
    }

/***********************************************Parking****/
    /**
     * Remove the nodes that where not used after running the algorithm
     * @throws Exception
     *

    protected void removeUnusedNodes() throws Exception{

    //Create and EditableBayesNet
    Instances instSmall = new Instances(inst);

    //Remove Unused Attributes
    for(int iNode = m_BayesNet.getNrOfNodes()-1; iNode >= 0; iNode--) {
    if((!hasParents(m_BayesNet, iNode) || !hasChildren(m_BayesNet, iNode)) && iNode != targetNode){
    //System.out.println("instSmall.deleteAttribute("+iNode+")");
    instSmall.deleteAttributeAt(iNode);
    }
    }

    EditableBayesNet bn = new EditableBayesNet(instSmall);

    //Add all arcs of parent nodes
    for(int iNode = 0; iNode < bn.getNrOfNodes(); iNode++) {
    int nodeIndex = 0;
    //get node Index
    for(int i = 0; i < m_BayesNet.getNrOfNodes(); i++){
    if(m_BayesNet.getNodeName(i) == bn.getNodeName(iNode)){
    nodeIndex = i;
    break;
    }
    }
    if(hasParents(m_BayesNet, nodeIndex)){
    int[] parents = m_BayesNet.getParentSet(nodeIndex).getParents();
    for(int jNode = 0; jNode < m_BayesNet.getParentSet(nodeIndex).getNrOfParents(); jNode++){
    int newNodeIndex = 0;
    for(int i = 0; i < bn.getNrOfNodes(); i++){
    if(bn.getNodeName(i) == m_BayesNet.getNodeName(parents[jNode])){
    newNodeIndex = i;
    break;
    }
    }
    bn.addArc(newNodeIndex, iNode); // addArc(parent, child)
    }
    }
    }

    //printFull(bn);
    m_BayesNet = bn;
    }*/





    /**
     * Performs a greedy, forward stepping search for the highest scoring model,
     * according to the score that is returned by ScoreNode.
     *
     * @param node is the target node
     */
    public void searchParents(int node) {
        boolean allTrue;
        double bestScore, score;
        int bestParent, parent;

        parents[node][0] = 0;
        System.out.println(String.format("=========== score: %f", scoreNode(node)));
        nodeScore = scoreNode(node) + prior_EBMC1(node);  // Score the node with no parents.
        numberOfModelsScored++;
        System.out.println(String.format("score: %10.5f time: %11.4f mins", nodeScore, timeMinutes()));
        while (parents[node][0] < maxParents) {
            bestScore = 1;
            bestParent = 0;
            allTrue = true;
            System.out.println(String.format("starting search for an additional predictor at %11.4f mins ...", timeMinutes()));

            // The last conjunct above insures that the first parent of the target will  not already be a child of the target.
            // This condition guarantees that the predictors (parents) of the target will be a unique set.
            // Otherwise, the search could get stuck in an infinite loop.
            for (int i = 1; i <= numberOfNodes; i++) {
                if ((i != node) && !parentPresent[i] && !childPresent[i]) {
                    allTrue = false;
                    parent = i;
                    parents[node][0]++;
                    parents[node][parents[node][0]] = parent;
                    score = scoreNode(node) + prior_EBMC1(node);
                    numberOfModelsScored++;
                    parents[node][0]--;

                    if ((score > bestScore) || (bestScore == 1)) {
                        bestParent = parent;
                        bestScore = score;
                    }
                }
            }

            if (allTrue) {
                break;
            }

            if (bestScore > nodeScore) {
                nodeScore = bestScore;
                parentPresent[bestParent] = true;
                parents[node][0]++;
                parents[node][parents[node][0]] = bestParent;
                StringBuilder info = new StringBuilder("current new predictors: ");
                for (int j = 1; j <= parents[node][0]; j++) {
                    info.append(String.format("%s ", nodeInfo[parents[node][j]].name)); //modify nodeInfo part
                }
                info.append(String.format(" score: %10.5f time: %11.4f mins", nodeScore, timeMinutes()));
                System.out.println(info.toString());
            } else {
                break;
            }
        }  // end of while-loop

        // reset to all false for future uses
        for (int i = 1; i <= parents[node][0]; i++) {
            parentPresent[parents[node][i]] = false;
        }
    }

    //did not change parent[][]
    protected void initVariables2(){

        readNodeInfo(numberOfNodes);
        maxParents = getMaxNrOfParents();
        maxChildren = getMaxNrOfChildren();

        maxValue = getMaxValue();
        maxCell = maxParents * maxValue * numberOfCases;
        map = new int[numberOfNodes + 1];
        values = new int[numberOfNodes + 1];

        newChildren = new int[MAXNEWCHILDREN + 1];

        children = new int[maxChildren + 1];
        nodeProb = new double[maxValue + 1];
        targetCounts = new int[maxValue + 1];

        childPresent = new boolean[numberOfNodes + 1];
        parentPresent = new boolean[numberOfNodes + 1];
        for (int i = 1; i <= numberOfNodes; i++) {
            childPresent[i] = false;
            parentPresent[i] = false;
            //	parents[i][0] = 0;
        }

        fileCaseCache = new FileCaseRecord[maxChildren + 1];
        for (int i = 1; i <= maxChildren; i++)
            fileCaseCache[i] = new FileCaseRecord(maxCell);

        counts = new int[maxCell + 1];
        countsTree = new int[maxCell + 1];
        lnChildProb = new double[numberOfCases + 1][maxValue + 1][maxChildren + 1];

        numberOfModelsScored = 0;
        numberOfChildren = 0;

        firstCase = lowerBound = 1;
        lastCase = upperBound = numberOfCases;

        targetNode = numberOfNodes;

        lnChildrenProb = new double[numberOfCases + 1][maxValue + 1];
        for (int casei = 1; casei <= lastCase; casei++)
            for (int j = 1; j <= nodeDimension[targetNode]; j++)
                lnChildrenProb[casei][j] = 0;
    }

    /**********************************************************Ye Parking code **************/
    //log. structure prior
    //source_clean_BayesNet, name match...
    //parent[][]
    public double OLDlogStructurePrior() {
        System.out.println("****current candidate:");
        System.out.println(printArray(parents));
//	    	System.out.println(source_clean_BayesNet);
        double k = 1.0/(1+ adjustedPriorSampleSize);
        //	System.out.println("m_PriorEquivalentSampleSize "+m_PriorEquivalentSampleSize);
        int sizeParentDiff = 0;
        //Save names of nodes in source_clean_BayesNet
        ArrayList<String> nodesInSourceClean = new ArrayList<String>();
        //For source_clean_BayesNet, this table saves a node's name and its index
        HashMap<String,Integer> sourceNodesNameIndexTable = new HashMap<String,Integer>();
        Instances sourceCleanInstances = source_clean_BayesNet.m_Instances;
        for (int jNode=0; jNode<sourceCleanInstances.numAttributes(); jNode++){
            String jNodeName = sourceCleanInstances.attribute(jNode).name();
            nodesInSourceClean.add(jNodeName);
            sourceNodesNameIndexTable.put(jNodeName, jNode);
        }
        for (int iNode=1; iNode<=numberOfNodes; iNode++){
            String iNodeName = nodeInfo[iNode].name;
            //	System.out.println("iNode:"+iNode+" "+iNodeName);
            if (nodesInSourceClean.contains(iNodeName)){
                //		System.out.println("iNode:"+iNode+" "+iNodeName);
                List<String> parentSetInSource = new ArrayList<String>();
                int iNodeIndexInSource = sourceNodesNameIndexTable.get(iNodeName);
                //		System.out.println(iNodeIndexInSource+"");
                ParentSet temp = source_clean_BayesNet.getParentSet(iNodeIndexInSource);
                //		System.out.println("number of parents:"+temp.getNrOfParents());
                int[] parentsInSourceClean = temp.getParents();
                //		System.out.println("parents in source clean:");
                for (int i=0; i<temp.getNrOfParents(); i++){
                    int indexParentInSource = parentsInSourceClean[i];
                    //			System.out.println("parent:"+sourceCleanInstances.attribute(indexParentInSource).name());
                    parentSetInSource.add(sourceCleanInstances.attribute(indexParentInSource).name());
                }
                //		System.out.println("parents in candidate:");
                List<String> parentSetInCandiate = new ArrayList<String>();
                int[] parentsInCandidate = parents[iNode];
                int numberOfParentInCandidate = parentsInCandidate[0];
                //		System.out.println("number of parents:"+numberOfParentInCandidate);
                for (int i=1; i<=numberOfParentInCandidate; i++){
                    int indexParentInCandidate = parentsInCandidate[i];
                    //			System.out.println(indexParentInCandidate+"");
                    String theParent = nodeInfo[indexParentInCandidate].name;
                    //			System.out.println("		" + theParent);
                    if (nodesInSourceClean.contains(theParent)){
                        parentSetInCandiate.add(theParent);
                        //				System.out.println("		add" + theParent);
                    }
                }
                //		System.out.println("parent set in source:");
                printList(parentSetInSource);
                //		System.out.println("parent set in candidate:");
                printList(parentSetInCandiate);
                sizeParentDiff  = sizeParentDiff + difference(parentSetInSource, parentSetInCandiate);
            }
        }
        //    	System.out.println("K:" + k);
        System.out.println("sizeParentDiff:" + sizeParentDiff);
        System.out.println("structure score:" + sizeParentDiff * Math.log(k));
        return sizeParentDiff * Math.log(k) ;
    }
    /**
     * Initiate with source model
     * @param predictors
     * @param max_parents
     * @param max_children
     * @param sourceModelFile
     * @throws Exception
     */
    //Ye added
    public BNTL(int predictors, int max_parents, int max_children, String sourceModelFile) throws Exception {
        setExpectedParentsOfTarget(predictors);
        setMaxNrOfParents(max_parents);
        setMaxNrOfChildren(max_children);
        source_BayesNet = new BIFReader();
        try {
            ((BIFReader) source_BayesNet).processFile(sourceModelFile);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        System.out.println("********************Original source model:");
        //	System.out.println(print(source_BayesNet));
        //	System.out.println(source_BayesNet);

    } // c'tor

    /**
     * Initiate with source model
     * @param predictors
     * @param max_parents
     * @param max_children
     * @param sourceModelFile
     * @throws Exception
     */
    //Ye added
    public BNTL(int predictors, int max_parents, int max_children, String sourceModelFile, int sourceSize) throws Exception {
        setExpectedParentsOfTarget(predictors);
        setMaxNrOfParents(max_parents);
        setMaxNrOfChildren(max_children);
        source_BayesNet = new BIFReader();
        try {
            ((BIFReader) source_BayesNet).processFile(sourceModelFile);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        m_SourceSampleSize = sourceSize;
        System.out.println("********************Original source model:");
        System.out.println(print(source_BayesNet));
        System.out.println(source_BayesNet);

    } // c'tor


    /**
     * inject target specific features into a bayesnetwork, use prevalence in target data as prob.
     * @param bayesNet
     * @param data
     * @return
     * @throws Exception
     */
    public BayesNet OLDinjectTargetSpecificFeature(BayesNet bayesNet, Instances data) throws Exception{
        System.out.println("********************Inject target specific features");
        PrintWriter printout = new PrintWriter(new File(utilityLoc+temporaryFileName));
        printout.println(bayesNet.graph());
        printout.flush(); printout.close();
        BIFReader reader = new BIFReader();
        EditableBayesNet tempNet = new EditableBayesNet(reader.processFile(utilityLoc+temporaryFileName));
        ArrayList<String> nodeList = new ArrayList<String>();
        int numNodes= tempNet.getNrOfNodes();
        for (int iNode=0; iNode<numNodes; iNode++){
            String nodeName =  tempNet.getNodeName(iNode);
            if (!nodeList.contains(nodeName)){
                nodeList.add(nodeName);
            }
        }
        //get a prob table from target data
        HashMap<String, HashMap<String,Double>> table = getProb(data);
        int numAttribute = data.numAttributes();
        for (int i=0; i<numAttribute; i++){
            Attribute oneAttribute = data.attribute(i);
            String candidateFeature = oneAttribute.name();
            //If a target feature is not in node list, then inject it into the bayesian network
            //change name, use the prob in target data as CPT for injected features.
            if (!nodeList.contains(candidateFeature)){
                tempNet.addNode(candidateFeature, oneAttribute.numValues());
                HashMap<String,Double> attributeProbTable = table.get(candidateFeature);
                String[] valueList = attributeProbTable.keySet().toArray(new String[0]);
                double[][] prob = new double[1][valueList.length];
                for (int j=0; j<valueList.length; j++){
                    String name = valueList[j];
                    int num = j+1;
                    String oldName = "Value" + num;
                    tempNet.renameNodeValue(tempNet.getNode(candidateFeature), oldName, name);
                    prob[0][j] = attributeProbTable.get(name);
                }
                tempNet.setDistribution(candidateFeature, prob);
                System.out.println("inject:" + candidateFeature);
            }
        }
        return tempNet;
    }



    /**
     * For iNode in oneNet, this method returns the corresponding index in anotherNet.
     * @param oneNet
     * @param iNode
     * @param anotherNet
     * @return
     */
    private static int getIndexNodeForAnotherNet(BayesNet oneNet, int iNode, BayesNet anotherNet){
        String iNodeName = oneNet.m_Instances.attribute(iNode).name();
        Instances instancesAnotherNet = anotherNet.m_Instances;
        for (int index=0; index<instancesAnotherNet.numAttributes(); index++){
            String name = instancesAnotherNet.attribute(index).name();
            if (iNodeName.equals(name)){
                return index;
            }
        }
        return -1;
    }

}

