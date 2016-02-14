/**
 * Created by Mostafa on 1/21/2016.
 */


import weka.clusterers.*;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils;
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

    public CustomClusters doKMeans(String path) throws Exception {
        SimpleKMeans cl = new SimpleKMeans();
        cl.setPreserveInstancesOrder(true);
        cl.setNumClusters(clusterCount);
        //cl.setSeed(10);
        return doCluster(path, cl);
    }

    public CustomClusters doHierarchical(String path) throws Exception {
        HierarchicalClusterer cl = new HierarchicalClusterer();
        //cl.setNumClusters(DataSource.read(path).size()-1);
        cl.setNumClusters(clusterCount);
        //cl.setDebug(true);
        //cl.setMaxIteration(500);
        //cl.setDistanceFunction(new ManhattanDistance());
        cl.setLinkType(new SelectedTag("AVERAGE", HierarchicalClusterer.TAGS_LINK_TYPE));
        return doCluster(path, cl);

    }

    public CustomClusters doHAK(String path) throws Exception {
        HAKClusterer cl = new HAKClusterer();
        cl.setPreserveInstancesOrder(true);
        //cl.setNumClusters(DataSource.read(path).size()-4);
        cl.setNumClusters(clusterCount);
        //cl.setDebug(true);
        cl.setMaxIteration(100);
        //cl.setDistanceFunction(new ManhattanDistance());
        cl.setLinkType(new SelectedTag("AVERAGE", HierarchicalClusterer.TAGS_LINK_TYPE));
        //cl.setSeed(3);
        cl.setMinimumInstanceCount(2);
        return doCluster(path, cl);
    }

    private CustomClusters doCluster(String filename, AbstractClusterer cl) throws Exception {


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
        CustomClusters cc = new CustomClusters(dataClusterer, cl, 1,getEvaluator(cl,data));
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
        List<CustomClusters> results = new ArrayList<CustomClusters>();
        String path = "C:\\Program Files\\Weka-3-6\\data\\diabetes.arff";
        ClusteringDemo clusteringDemo = new ClusteringDemo();
        //clusteringDemo.ClusterWithSimpleKMeans(path);
        //clusteringDemo.classToCluster(path);
        results.add(clusteringDemo.doKMeans(path));
        //clusteringDemo.doHierarchical(path);
        results.add(clusteringDemo.doHAK(path));
        //clusteringDemo.visualize(path);
        //clusteringDemo.visualizeClusterAssignments(path);

        StringBuilder sb = new StringBuilder();
        for (CustomClusters result : results) {
            sb.append(result.getResultString());
        }
        System.out.println(sb.toString());
    }

    private int clusterCount = 2;

}
