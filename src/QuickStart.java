public class QuickStart {

    public static void main(String[] args) throws Exception {

        // configuration in IntellIDEA

        // Edit Configuration---> Modify Options(blue letter)--->Add VM Options-->
        // add this -Djava.library.path= <your absolute path>/lib

        //[For example ]
        // -Djava.library.path=/Users/johnsong/Documents/Ye_lab/releaseVersion/lib

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
