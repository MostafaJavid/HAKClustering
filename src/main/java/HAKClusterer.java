import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.SelectedTag;

import java.util.*;

/**
 * Created by Mostafa on 1/25/2016.
 */
public class HAKClusterer extends SimpleKMeans {
    int maxIteration = Integer.MAX_VALUE;
    int m_nLinkType = 0;//SINGLE
    private int minimumInstanceCount = 2;

    @Override
    public void buildClusterer(Instances data) throws Exception {
        HierarchicalClusterer hierarchicalClusterer = getHierarchicalClusterer(data.size());
        hierarchicalClusterer.buildClusterer(data);

        Instances centroids = getCentroids(data, hierarchicalClusterer);
        super.setInitializationMethod(new SelectedTag("HAK", SimpleKMeans.TAGS_SELECTION));
        super.setHakCentroids(centroids);
        super.buildClusterer(data);
        //super.get
    }

//////////////////////METHODS////////////////////////////////////////////////////////////////////////

    public HierarchicalClusterer getHierarchicalClusterer(int size) throws Exception {
        HierarchicalClusterer cl = new HierarchicalClusterer();
        cl.setNumClusters(size - getMaxIteration());
        cl.setDebug(getDebug());
        cl.setMaxIteration(getMaxIteration());
//        //cl.setDistanceFunction(new ManhattanDistance());
        cl.setLinkType(getLinkType());
        //cl.clusterInstance()
        return cl;
    }

    private Instances getCentroids(Instances data, HierarchicalClusterer hierarchicalClusterer) {
        CustomClusters customClusters = new CustomClusters(data, hierarchicalClusterer, getMinimumInstanceCount());
        return customClusters.getCentroidsAsInstances();
    }

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
}
