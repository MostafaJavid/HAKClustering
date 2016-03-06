/**
 * Created by Mostafa on 1/21/2016.
 */


import weka.clusterers.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.explorer.ClustererAssignmentsPlotInstances;
import weka.gui.explorer.ExplorerDefaults;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

import javax.swing.*;
import java.awt.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

class ClusteringDemo {

    public void ClusterWithEM(String filename) throws Exception {
        ClusterEvaluation eval;
        Instances data;
        String[] options;
        DensityBasedClusterer cl;
        double logLikelyhood;

        data = DataSource.read(filename);

        // normal
        System.out.println("\n--> normal");
        options = new String[2];
        options[0] = "-t";
        options[1] = filename;
        System.out.println(ClusterEvaluation.evaluateClusterer(new EM(), options));

        // manual call
        System.out.println("\n--> manual");
        cl = new EM();
        cl.buildClusterer(data);
        eval = new ClusterEvaluation();
        eval.setClusterer(cl);
        eval.evaluateClusterer(new Instances(data));
        System.out.println(eval.clusterResultsToString());

        // cross-validation for density based clusterers
        // NB: use MakeDensityBasedClusterer to turn any non-density clusterer
        //     into such.
        System.out.println("\n--> Cross-validation");
        cl = new EM();
        logLikelyhood = ClusterEvaluation.crossValidateModel(
                cl, data, 10, data.getRandomNumberGenerator(1));
        System.out.println("log-likelyhood: " + logLikelyhood);
    }

    public void visualize(String path) throws Exception {
        // load data
        Instances train = DataSource.read(path);
        // some data formats store the class attribute information as well
        if (train.classIndex() != -1)
            throw new IllegalArgumentException("Data cannot have class attribute!");

        // instantiate clusterer
//        String[] options = Utils.splitOptions(Utils.getOption('W', args));
//        String classname = options[0];
//        options[0] = "";
        SimpleKMeans clusterer = new SimpleKMeans();  //AbstractClusterer.forName(classname, options);
        clusterer.setNumClusters(3);

        // evaluate clusterer
        clusterer.buildClusterer(train);
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(train);

        // setup visualization
        // taken from: ClustererPanel.startClusterer()
        ClustererAssignmentsPlotInstances plotInstances = new ClustererAssignmentsPlotInstances();
        plotInstances.setClusterer(clusterer);
        plotInstances.setInstances(train);
        plotInstances.setClusterEvaluation(eval);
        plotInstances.setUp();
        String name = (new SimpleDateFormat("HH:mm:ss - ")).format(new Date());
        String cname = clusterer.getClass().getName();
        if (cname.startsWith("weka.clusterers."))
            name += cname.substring("weka.clusterers.".length());
        else
            name += cname;
        name = name + " (" + train.relationName() + ")";
        VisualizePanel vp = new VisualizePanel();
        vp.setName(name);
        vp.addPlot(plotInstances.getPlotData(cname));

        // display data
        // taken from: ClustererPanel.visualizeClusterAssignments(VisualizePanel)
        JFrame jf = new JFrame("Weka Clusterer Visualize: " + vp.getName());
        jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        jf.setSize(500, 400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vp, BorderLayout.CENTER);
        jf.setVisible(true);
    }

    public void visualizeClusterAssignments(String path) throws Exception {
        // load data
        Instances train = DataSource.read(path);
        // some data formats store the class attribute information as well
        if (train.classIndex() != -1)
            throw new IllegalArgumentException("Data cannot have class attribute!");

        // instantiate clusterer
//        String[] options = Utils.splitOptions(Utils.getOption('W', args));
//        String classname = options[0];
//        options[0] = "";
//        Clusterer clusterer = AbstractClusterer.forName(classname, options);

        SimpleKMeans clusterer = new SimpleKMeans();
        clusterer.setNumClusters(3);

        // evaluate clusterer
        clusterer.buildClusterer(train);
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(train);

        // setup visualization
        // taken from: ClustererPanel.startClusterer()
        ClustererAssignmentsPlotInstances plotInstances = ExplorerDefaults.getClustererAssignmentsPlotInstances();
        plotInstances.setClusterer(clusterer);
        plotInstances.setInstances(train);
        plotInstances.setClusterEvaluation(eval);
        plotInstances.setUp();
        String name = (new SimpleDateFormat("HH:mm:ss - ")).format(new Date());
        String cname = clusterer.getClass().getName();
        if (cname.startsWith("weka.clusterers."))
            name += cname.substring("weka.clusterers.".length());
        else
            name += cname;
        PlotData2D predData = plotInstances.getPlotData(name);

        VisualizePanel vp = new VisualizePanel();
        vp.setName(predData.getPlotName());
        vp.addPlot(predData);

        // display data
        // taken from: ClustererPanel.visualizeClusterAssignments(VisualizePanel)
        String plotName = vp.getName();
        JFrame jf = new JFrame("Weka Clusterer Visualize: " + plotName);
        jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jf.setSize(500, 400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vp, BorderLayout.CENTER);
        jf.setVisible(true);
    }

    public CustomClusters doKMeans(String title,String path, int clusterCount) throws Exception {
        SimpleKMeans cl = new SimpleKMeans();
        cl.setPreserveInstancesOrder(true);
        cl.setNumClusters(clusterCount);
        //cl.setSeed(10);
        return doCluster(title,path, cl);
    }

    public CustomClusters doHierarchical(String title,String path, int clusterCount, LinkType linkType) throws Exception {
        HierarchicalClusterer cl = new HierarchicalClusterer();
        //cl.setNumClusters(DataSource.read(path).size()-1);
        cl.setNumClusters(clusterCount);
        //cl.setDebug(true);
        //cl.setMaxIteration(500);
        //cl.setDistanceFunction(new ManhattanDistance());
        cl.setLinkType(linkType.getTag());
        return doCluster(title,path, cl);

    }

    public CustomClusters doHAK(String title,String path, int clusterCount, LinkType linkType, int maxIteration, int minInstanceCount,int outlierFactor,int outlierMinDense) throws Exception {
        HAKClusterer cl = new HAKClusterer();
        cl.setPreserveInstancesOrder(true);
        //cl.setNumClusters(DataSource.read(path).size()-4);
        cl.setNumClusters(clusterCount);
        cl.setDebug(true);
        cl.setMaxIteration(maxIteration);
        //cl.setDistanceFunction(new ManhattanDistance());
        cl.setLinkType(linkType.getTag());
        //cl.setSeed(3);
        cl.setMinimumInstanceCount(minInstanceCount);
        cl.setOutlierFactor(outlierFactor);
        cl.setOutlierMinDense(outlierMinDense);

        Instances data = readData(path);
        Instances dataClusterer = filterClassAttribute(data);
        cl.buildClusterer(dataClusterer);
        cl.removeOutlier(data);
        CustomClusters cc = new CustomClusters(title,dataClusterer, cl, 1, getEvaluator(cl, data));
        return cc;
    }

    private CustomClusters doCluster(String title,String filename, AbstractClusterer cl) throws Exception {


//        String[] options;
//        double logLikelyhood;

        Instances data = readData(filename);
        Instances dataClusterer = filterClassAttribute(data);


        // normal
//        System.out.println("\n--> normal");
//        options = new String[2];
//        options[0] = "-t";
//        options[1] = filename;
        //System.out.println(ClusterEvaluation.evaluateClusterer(new SimpleKMeans(), options));

        // manual call
        //System.out.println("\n--> manual");
        cl.buildClusterer(dataClusterer);

        CustomClusters cc = new CustomClusters(title,dataClusterer, cl, 1, getEvaluator(cl, data));
        //System.out.println(cc.getResultString());
        return cc;


        // cross-validation for density based clusterers
        // NB: use MakeDensityBasedClusterer to turn any non-density clusterer
        //     into such.
//        System.out.println("\n--> Cross-validation");
//        cl = new EM();
//        logLikelyhood = ClusterEvaluation.crossValidateModel(
//                cl, data, 10, data.getRandomNumberGenerator(1));
//        System.out.println("log-likelyhood: " + logLikelyhood);
    }

    private ClusterEvaluation getEvaluator(AbstractClusterer cl, Instances data) throws Exception {
        ClusterEvaluation eval;
        eval = new ClusterEvaluation();
        eval.setClusterer(cl);
        eval.evaluateClusterer(data);
        //System.out.println(eval.clusterResultsToString());
        return eval;
    }

    private Instances readData(String filename) throws Exception {
        Instances data;
        data = DataSource.read(filename);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    private Instances filterClassAttribute(Instances data) throws Exception {
        // generate data for clusterer (w/o class)
        Remove filter = new Remove();
        filter.setAttributeIndices("" + (data.classIndex() + 1));
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }

    /**
     * usage:
     * ClusteringDemo arff-file
     */
    public static void main(String[] args) throws Exception {
        AVGResultBeautifier results = new AVGResultBeautifier();
        LinkType linkType = LinkType.SINGLE;
        results.add(getResult("iris",3,linkType,15,2,2));//150
        results.add(getResult("diabetes",2,linkType,76,7,2));//768
        results.add(getResult("glass",7,linkType,40,10,2));//214
        results.add(getResult("ionosphere",2,linkType,120,15,2));//351
        results.add(getResult("segment-challenge",7,linkType,150,20,2));//1500-100,10,3
        results.add(getResult("segment-test",7,linkType,100,10,3));//810,100,10,3
        results.add(getResult("unbalanced",2,linkType,30,4,3));//856
        System.out.println(results.printAllResults());
    }

    private static ResultBeautifier getResult(String fileName,int clusterCount,LinkType linkType,int maxIteration,int outlierFactor,int outlierMinDense) throws Exception {
        List<CustomClusters> results = new ArrayList<CustomClusters>();
        String path = "C:\\Program Files\\Weka-3-6\\data\\" + fileName+".arff";
        ClusteringDemo clusteringDemo = new ClusteringDemo();
        //clusteringDemo.ClusterWithSimpleKMeans(path);
        //clusteringDemo.classToCluster(path);
        results.add(clusteringDemo.doKMeans("KMeans",path, clusterCount));
//        results.add(clusteringDemo.doHierarchical("Hierarchical",path, clusterCount, LinkType.SINGLE));
//        results.add(clusteringDemo.doHAK("HAK-CENTROID",path, clusterCount, LinkType.CENTROID, maxIteration, 2,outlierFactor,outlierMinDense));
        results.add(clusteringDemo.doHAK("HAK-SINGLE",path, clusterCount, LinkType.SINGLE, maxIteration, 2,outlierFactor,outlierMinDense));
//        results.add(clusteringDemo.doHAK("HAK-COMPLETE",path, clusterCount, LinkType.COMPLETE, maxIteration, 2,outlierFactor,outlierMinDense));
//        results.add(clusteringDemo.doHAK("HAK-AVERAGE",path, clusterCount, LinkType.AVERAGE, maxIteration, 2,outlierFactor,outlierMinDense));
//        //clusteringDemo.visualize(path);
        //clusteringDemo.visualizeClusterAssignments(path);

        return new ResultBeautifier(fileName,results);
    }


}
