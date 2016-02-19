import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

import java.util.*;
import java.util.List;

/**
 * Created by Mostafa on 1/25/2016.
 */
public class HAKClusterer extends SimpleKMeans {
    int maxIteration = Integer.MAX_VALUE;
    int m_nLinkType = 0;//SINGLE
    private int minimumInstanceCount = 2;
    private int outlierFactor = 5;
    private int outlierMinDense = 3;
    Instances centroids;
    CustomClusters middleClusters;
    CustomClusters middleClusters2;
    Map<Integer,Integer> outliers;

    @Override
    public void buildClusterer(Instances data) throws Exception {
        HierarchicalClusterer hierarchicalClusterer = getHierarchicalClusterer(getMaxIteration());
        hierarchicalClusterer.buildClusterer(data);
        outliers = getOutliers(hierarchicalClusterer);

        middleClusters = new CustomClusters("middleClusters",data, hierarchicalClusterer, minimumInstanceCount, null);
        Instances middleInstances = middleClusters.getCentroidsAsInstances();
        HierarchicalClusterer hierarchicalClusterer2 = getHierarchicalClusterer(middleInstances.size());
        hierarchicalClusterer2.buildClusterer(middleInstances);
        middleClusters2 = new CustomClusters("middleClusters2",middleInstances, hierarchicalClusterer2, 1, null);
        centroids = middleClusters2.getCentroidsAsInstances();

        super.setInitializationMethod(new SelectedTag("HAK", SimpleKMeans.TAGS_SELECTION));
        super.setHakCentroids(centroids);
        List<Instance> outlierInstances = new ArrayList<Instance>();
        for (Integer outlier : outliers.keySet()) {
            outlierInstances.add(data.instance(outlier));
        }
        data.removeAll(outlierInstances);
        super.buildClusterer(data);
    }

    private Map<Integer,Integer> getOutliers(HierarchicalClusterer hierarchicalClusterer) {
        double[][] distances = hierarchicalClusterer.getDistances();
        double avgDistance = hierarchicalClusterer.getAvgDistanceOfIterations();
        double maxDistance = avgDistance * getOutlierFactor();
        Map<Integer, Integer> outliers = new HashMap<Integer, Integer>();
        for (int i = 0; i < distances.length; i++) {
            int count = 0;
            for (int j = 0; j < distances[i].length; j++) {
                if (distances[i][j] < maxDistance) {
                    count++;
                }
            }
            if (count < getOutlierMinDense()) {
                outliers.put(i, count);
            }
        }

        return outliers;
    }

//////////////////////METHODS////////////////////////////////////////////////////////////////////////

    public HierarchicalClusterer getHierarchicalClusterer(int maxIteration) throws Exception {
        HierarchicalClusterer cl = new HierarchicalClusterer();
        //cl.setNumClusters(size - getMaxIteration());
        cl.setNumClusters(getNumClusters());
        cl.setDebug(getDebug());
        cl.setMaxIteration(maxIteration);
//        //cl.setDistanceFunction(new ManhattanDistance());
        cl.setLinkType(getLinkType());
        //cl.clusterInstance()
        return cl;
    }

//    private Instances getCentroids(Instances data, HierarchicalClusterer hierarchicalClusterer, int minimumInstanceCount) {
//        middleClusters = new CustomClusters(data, hierarchicalClusterer, minimumInstanceCount, null);
//        //customClusters.filterClusters(getNumClusters());
//        return middleClusters.getCentroidsAsInstances();
//        //List<CustomCluster> list = new ArrayList<CustomCluster>(Arrays.asList(customClusters.getCustomClusterArray()));
//        //list = filterClusters(list,getNumClusters());
//        //return CustomClusters.convertToInstances(list,data);
//    }


/////////////////////PROPS/////////////////////////////////////////////////////////////////////

    public void setLinkType(SelectedTag newLinkType) {
        if (newLinkType.getTags() == HierarchicalClusterer.TAGS_LINK_TYPE) {
            m_nLinkType = newLinkType.getSelectedTag().getID();
        }
    }

    public SelectedTag getLinkType() {
        return new SelectedTag(m_nLinkType, HierarchicalClusterer.TAGS_LINK_TYPE);
    }

    public int getMaxIteration() {
        return maxIteration;
    }

    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }

    public int getMinimumInstanceCount() {
        return minimumInstanceCount;
    }

    public void setMinimumInstanceCount(int minimumInstanceCount) {
        this.minimumInstanceCount = minimumInstanceCount;
    }

    public CustomClusters getMiddleClusters() {
        return middleClusters;
    }

    public CustomClusters getMiddleClusters2() {
        return middleClusters2;
    }

    public int getOutlierFactor() {
        return outlierFactor;
    }

    public void setOutlierFactor(int outlierFactor) {
        this.outlierFactor = outlierFactor;
    }

    public int getOutlierMinDense() {
        return outlierMinDense;
    }

    public void setOutlierMinDense(int outlierMinDense) {
        this.outlierMinDense = outlierMinDense;
    }

    public Map<Integer, Integer> getOutliers() {
        return outliers;
    }
}
