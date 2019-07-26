using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;

namespace FFNN
{
    /// <summary>
    /// Read in the data file and return corresponding DataTable after performing necessary operations.
    /// </summary>
    public class DataFileOperations
    {
        #region Properties

        // Complete table property.
        public DataTable CompleteTable { get; internal set; }

        // Train table property.
        public DataTable TrainTable { get; internal set; }

        // Test table property.
        public DataTable TestTable { get; internal set; }

        #endregion

        #region Public Methods

        // Creates the Train and Test collections using the provided text document.
        public void CreateTrainAndTestSets()
        {
            // Temporary variables.
            var aRand = new Random();
            var aCompleteRandomTable = new DataTable();
            for (int col = 0; col < CompleteTable.Columns.Count; col++)
            {
                aCompleteRandomTable.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));
            }

            // Add designated size of train set data points from complete set to train table. Completely randomized.
            aCompleteRandomTable = CompleteTable.AsEnumerable().OrderBy(r => aRand.Next()).Take(CompleteTable.Rows.Count).CopyToDataTable();

            int rowNumber = 0;
            foreach (DataRow row in aCompleteRandomTable.Rows)
            {
                if (rowNumber < 4000)
                {
                    TrainTable.ImportRow(row);
                }
                else
                {
                    TestTable.ImportRow(row);
                }

                rowNumber++;
            }
        }

        // Take each entry in a datatable locally hosted and pass to .csv doc with tab delimiter.
        public void ConvertDataTableToTextFile(DataTable theDt, string theFileName)
        {
            var builder = new StringBuilder();

            foreach (DataRow row in theDt.Rows)
            {
                foreach (var cell in row.ItemArray)
                {
                    builder.Append(cell.ToString());

                    if (cell != row.ItemArray.Last())
                    {
                        builder.Append("\t");
                    }
                }

                builder.Append(Environment.NewLine);
            }

            var file = new FileStream(theFileName, FileMode.Create);
            var writer = new StreamWriter(file);

            writer.Write(builder);
            writer.Flush();
            writer.Close();
        }

        // Converts the designated MNISTnumImages5000 to 5000x748 table. Tab deliminator designated.
        public void ConvertTextToDataTable(int numberOfColumns)
        {
            // Get current directory filepath and designated file name statically.
            string filePath = Path.Combine(Directory.GetCurrentDirectory(), "MNISTnumImages5000.txt");

            // Create data table and provide designated number of columns.
            DataTable tbl = new DataTable();
            for (int col = 0; col < numberOfColumns; col++)
                tbl.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));

            // Read file lines.
            string[] lines = File.ReadAllLines(filePath);

            // Populate row until total number of columns populated.
            foreach (string line in lines)
            {
                var cols = line.Split('\t');

                DataRow dr = tbl.NewRow();
                for (int cIndex = 0; cIndex < numberOfColumns-2; cIndex++)
                {
                    dr[cIndex] = cols[cIndex];
                }

                tbl.Rows.Add(dr);
            }

            AddBiasesToDataTable(tbl);
        }

        // Overload method for use with pregenerated .txt files.
        public DataTable ConvertTextToDataTable(int numberOfColumns, string theFileName)
        {
            // Get current directory filepath and designated file name statically.
            string filePath = Path.Combine(Directory.GetCurrentDirectory(), theFileName);

            // Create data table and provide designated number of columns.
            DataTable tbl = new DataTable();
            for (int col = 0; col < numberOfColumns; col++)
                tbl.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));

            // Read file lines.
            string[] lines = File.ReadAllLines(filePath);

            // Populate row until total number of columns populated.
            foreach (string line in lines)
            {
                var cols = line.Split('\t');

                DataRow dr = tbl.NewRow();
                for (int cIndex = 0; cIndex < numberOfColumns; cIndex++)
                {
                    dr[cIndex] = cols[cIndex];
                }

                tbl.Rows.Add(dr);
            }

            return tbl;
        }

        // Print a list to a designated text file.
        public void ConvertListToTextFile(List<double> theList, string fileName)
        {
            TextWriter tw = new StreamWriter(fileName);

            foreach (var item in theList)
            {
                tw.WriteLine(item);
            }

            tw.Close();
        }

        #endregion

        #region Private Methods

        // Adds the labels to the end of each row in a given datatable. 
        void AddLabelsToDataTable(DataTable theDt)
        {
            // Get current directory filepath and designated file name statically.
            string filePath = Path.Combine(Directory.GetCurrentDirectory(), "MNISTnumLabels5000.txt");

            // Read file lines.
            string[] lines = File.ReadAllLines(filePath);

            // Index to add item.
            int aIndex = 0;

            // Add label value to end of data table rows.
            foreach (string item in lines)
            {
                theDt.Rows[aIndex]["Column786"] = item;
                aIndex++;
            }

            // Update CompleteTable to correct value and copy dimensions for train and test tables.
            CompleteTable = theDt;
            TrainTable = CompleteTable.Clone();
            TestTable = CompleteTable.Clone();
        }

        void AddBiasesToDataTable(DataTable theDt)
        {
            // Initialize Random variable to remove time-dependent issues.
            Random theRandomDouble = new Random();

            // Add label value to end of data table rows.
            for (int i = 0; i < theDt.Rows.Count; i++)
            {
                theDt.Rows[i]["Column785"] = (theRandomDouble.NextDouble() * 2.0 - 1.0);
            }

            AddLabelsToDataTable(theDt);
        }

        #endregion
    }
}
