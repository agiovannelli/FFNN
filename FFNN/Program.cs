using System;

namespace FFNN
{
    class MainClass
    {
        public static void Main()
        {
            Console.WriteLine("Welcome to the Feed-forward backpropagation application!\n" +
                "This application was written in college, but has been refactored to include this CLI.\n\n" +
                "Would you like to perform data operations? This will create and train test data sets held locally.\n" +
                "\nType 'Y' or hit enter to continue: ");

            var performDataOperations = Console.ReadLine();

            // Instantiate classes to utilize methods.
            DataFileOperations theDtFileOps = new DataFileOperations();
            RubricProblems theRubricProblems = new RubricProblems(theDtFileOps);

            if (performDataOperations.ToLower().Equals("y"))
            {
                // Import complete data and create random train and test sets. Use 786 to have a column for bias at 785.
                Console.WriteLine("\nData set training selected.\nConverting text from document to data table...");
                theDtFileOps.ConvertTextToDataTable(786);

                Console.WriteLine("\nData table created. \n\nGenerating train and test data sets...");
                theDtFileOps.CreateTrainAndTestSets();

                // Export Train and Test tables respectively.
                Console.WriteLine("\nTrain and test data sets created. Writing complete table text file...");
                theDtFileOps.ConvertDataTableToTextFile(theDtFileOps.CompleteTable, "CompleteTable.txt");

                Console.WriteLine("\nComplete table file created. Writing train table text file...");
                theDtFileOps.ConvertDataTableToTextFile(theDtFileOps.TrainTable, "TrainTable.txt");

                Console.WriteLine("\nTrain table created. Writing test table text file...");
                theDtFileOps.ConvertDataTableToTextFile(theDtFileOps.TestTable, "TestTable.txt");

                Console.WriteLine("\nTest table file created.\n\nData file operations completed.");
            }

            Console.WriteLine("This application has been segmented into problems as required per course rubric.\n" +
                "Please enter the problem number you would like to run (1-4) or enter to exit: ");

            var problemValue = Console.ReadLine();

            switch (int.Parse(problemValue))
            {
                case 1:
                    Console.WriteLine("\nProblem one selected. Starting processing...");
                    theRubricProblems.ProblemOne();
                    Console.WriteLine("\nProblem one completed.");
                    break;
                case 2:
                    Console.WriteLine("\nProblem two selected. Starting processing...");
                    theRubricProblems.ProblemTwo();
                    Console.WriteLine("\nProblem two completed.");
                    break;
                case 3:
                    Console.WriteLine("\nProblem three selected. Starting processing...");
                    theRubricProblems.ProblemThree();
                    Console.WriteLine("\nProblem three completed.");
                    break;
                case 4:
                    Console.WriteLine("\nProblem four selected. Starting processing...");
                    theRubricProblems.ProblemFour();
                    Console.WriteLine("\nProblem four completed.");
                    break;
            }

            Console.WriteLine("\n\nExiting application...");
        }
    }
}