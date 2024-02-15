public class QuickStart {

    public static void main(String[] args) throws Exception {



        //-Djava.library.path=/Users/johnsong/Documents/Ye_lab/releaseVersion/lib

        // run example-------------------------------
        runnerBNTL runnerBNTL = new runnerBNTL();
        String[] res = {"./smile_license/License.java",
                "./no_KL",
                "FLU-SLC2AC-0809-transfer20090118-IG001-NB", // find this configuration file in no_KL--> utility-->
                "unadjust","ratio","off" //
        };

        runnerBNTL.main(res);

    }

}
