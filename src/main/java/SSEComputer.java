import weka.clusterers.HierarchicalClusterer;
import weka.core.Instances;

import java.util.List;
import java.util.Vector;

/**
 * Created by Mostafa on 2/1/2016.
 */
public class SSEComputer {
//    public void Compute(Instances data, HierarchicalClusterer hierarchicalClusterer){
//        List<CustomCluster> centroids = ClusteringHelper.getCentroids(data,hierarchicalClusterer,1);
//        double[] SSE = new double[hierarchicalClusterer.nClusterID.length];
//        for (int i = 0; i < hierarchicalClusterer.nClusterID.length; i++) {
//            Vector<Integer> cluster = hierarchicalClusterer.nClusterID[i];
//            if (cluster.size() > 0){
//                CustomCluster centroid = new CustomCluster();
//                centroid.setClusterId(i);
//                centroid.setClusterCount(cluster.size());
//                for (int j = 0; j < cluster.size(); j++) {
//                    int index = cluster.get(j);
//
//                }
//            }
//            else{
//                SSE[i] = -1;
//            }
//        }
//    }
}
