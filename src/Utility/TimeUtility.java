package Utility;

import org.joda.time.DateTime;

import org.joda.time.Interval;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

/**
 *
 * Sep 14, 2011
 * 3:34:26 PM
 * @author Kevin V. Bui (kvb2@pitt.edu)
 */
public class TimeUtility {

    public static final DateTimeFormatter formatter = DateTimeFormat.forPattern("MM/dd/yyyy hh:mm:ss a");


    private TimeUtility() { }

    /**
     * Computes the time interval between the start time and end time.
     *
     * @param start time work has started
     * @param end time work has ended
     * @return the time interval between start time and end time in milliseconds.
     */
    public static double timeSecondIntervalMillisec(DateTime start, DateTime end) {
        Interval interval = new Interval(start, end);

        return (double) interval.toDurationMillis();
    }

    /**
     * Computes the time interval between the start time and end time.
     *
     * @param start time work has started
     * @param end time work has ended
     * @return the time interval between start time and end time in milliseconds.
     */
    public static double getDurationInSeconds(DateTime start, DateTime end) {
        Interval interval = new Interval(start, end);

        return (double) interval.toDurationMillis() / 1000;
    }

    public static double getDurationInMinutes(DateTime start, DateTime end) {
        Interval interval = new Interval(start, end);

        return (double) interval.toDurationMillis() / 60000;
    }

}

