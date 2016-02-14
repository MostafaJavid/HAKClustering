import java.io.File;
import java.io.FileWriter;
import java.util.Enumeration;
import java.util.Formatter;
import java.util.Locale;
import java.util.Vector;

import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;


public class SilhouetteCoefficient extends SimpleKMeans {

    private static final long serialVersionUID = 5643670507674593214L;
    private final FileWriter file;

    StringBuilder clusteringResults = new StringBuilder();

    boolean kmeanPlusPlus=false;

    public SilhouetteCoefficient(FileWriter file) {
        this.file=file;
    }

    public void setKmeanPlusPlus(boolean kmeanPlusPlus) {
        this.kmeanPlusPlus = kmeanPlusPlus;
    }

    /**
     * Cluster data with k-means algorithm using the value k obtained by
     * computing the maximum silhouette coefficient.
     *
     * @param data data for clustering
     */
    public void buildClusterer(Instances data) throws Exception {
		/*amount of data*/
        int n = data.numInstances();    // Reduce n for testing

		/*KMeans Object*/
        SimpleKMeans kmeans = new SimpleKMeans();

        // best value for k
        int bestClustering = 2;
        // best silhouette coefficient
        double maxSil = Double.NEGATIVE_INFINITY;

        Formatter formatter = new Formatter(clusteringResults, Locale.US);

        // for every number of clusters
        //for (int k = 2; k < 5*n/10; k++) {    // max. 50 %
        for (int k = 2; k < Math.sqrt(n); k++) {    // max. 50 %
            kmeans.setNumClusters(k);
            String[] options1= new String[2];
            options1[0]="-init";
            if(kmeanPlusPlus) {
                options1[1] = "1";
            }
            else {
                options1[1]="0";
            }

            //SimpleKMeans kMeans=new SimpleKMeans();
            kmeans.setOptions(options1);
            kmeans.buildClusterer(data);
            assert k == kmeans.numberOfClusters();

            // compute silhouette coefficient
            double sil = 0.0;
            double[] clusterSize = kmeans.getClusterSizes();
            double[] clusterInstance = new double[n];

            // calculate cluster for each element
            for (int i = 0; i < n; i++)
                clusterInstance[i] = kmeans.clusterInstance(data.instance(i));
			
			
			/* start iteration for silhouette coefficient calculation ****** */
            double silSum = 0.;
            double compactness=0.0;
            double separation=0.0;

            // for each element
            for (int i = 0; i < n; i++) {
                Instance o = data.instance(i);
                double clusterOfO = clusterInstance[i];

			    /*
			     * calculate sum of distances between object o and all clusters
			     */

                double [] 	sumOfODistances = new double[k];

                for(int j = 0; j < n; j++){
                    sumOfODistances[(int)clusterInstance[j]] += kmeans.getDistanceFunction().distance(o, data.instance(j));
                }
				
				/*
				 * calculate average of distances between object o and all clusters 
				 * and determine the minimum average distance to other clusters
				 */

                double[] averageOfODistances = new double[k];

                double minAverage = Double.POSITIVE_INFINITY;
                for(int clusterNubmer = 0; clusterNubmer < clusterSize.length; clusterNubmer++){
                    averageOfODistances[clusterNubmer] = sumOfODistances[clusterNubmer] / clusterSize[clusterNubmer];
                    if(clusterNubmer != clusterOfO && averageOfODistances[clusterNubmer] < minAverage){
                        minAverage = averageOfODistances[clusterNubmer];
                    }
                }

                // calculate sum of sils
                sil=(minAverage - averageOfODistances[(int)clusterOfO]) /
                        Math.max(minAverage, averageOfODistances[(int)clusterOfO]);

                compactness += minAverage;
                separation += averageOfODistances[(int)clusterOfO];

                //System.out.println("Silhoutte Coefficient for Object O " + data.instance(i)+"="+sil);

                silSum += (minAverage - averageOfODistances[(int)clusterOfO]) /
                        Math.max(minAverage, averageOfODistances[(int)clusterOfO]);


            }

            // calculate average of sils
            sil = silSum / (double)n;
            System.out.println("------------------------------------------------------------");
            file.write("--------------------------------------------------------------------");

            System.out.println("compactness for " + k + " cluster is: " + compactness / (double) n);
            file.write("compactness for " + k + " cluster is: " + compactness / (double) n + "\n");
            System.out.println("separation for " + k + " cluster is: " + separation / (double) n);
            file.write("separation for " + k + " cluster is: " + separation / (double) n+"\n");
			
			/*
			 * determine maximum of sils
			 */

            if(sil > maxSil){
                maxSil = sil;
                bestClustering = k;
            }

            formatter.format("k = %3d, s = %5.3f\n", k, sil);
            System.out.printf("k = %3d, s = %5.3f\n", k, sil);
            System.out.println("number of clusters selected by k-mean: "+kmeans.numberOfClusters());
            System.out.println("number of clusters selected for assessing: "+k);

            //file.write("k = %3d, s = %5.3f\n", k, sil);
            file.write("number of clusters selected by k-mean: " + kmeans.numberOfClusters() + "\n");
            file.write("number of clusters selected for assessing: " + k + "\n");
            super.toString();
        }

        formatter.format("\nSilhouette Coefficient s = %5.3f (k = %d)\n",
                maxSil, bestClustering);
        super.setNumClusters(bestClustering);
        super.buildClusterer(data);
    }


    /**
     * return the final output
     *
     * @return description of the clusters and the result of the silhouette
     *         coefficient as a string
     */
    public String toString() {
        return clusteringResults.toString() + super.toString();
    }


    /**
     * No options available
     */
    @SuppressWarnings("unchecked")
    public Enumeration listOptions() {
        Vector result = new Vector();
        return result.elements();
    }


    public String[] getOptions() {
        String[] options = {};
        return options;
    }


    /**
     * @return a description of the evaluator suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "Silhouette coefficient for SimpleKMeans algorithm";
    }

}
