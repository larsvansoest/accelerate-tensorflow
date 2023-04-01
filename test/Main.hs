{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RebindableSyntax #-}

import Data.Array.Accelerate hiding (Eq, Vector)
import Prelude hiding (unzip3, unzip, zip, zipWith3, zip3, (<=), (>), filter, (&&), (>=), subtract, (<), truncate, (++), fromIntegral, map, zipWith, (+))
import Data.Accelerate.TensorFlow.Execute
import Data.Array.Accelerate.Interpreter

import Test.Tasty
import Test.Tasty.HUnit
import Data.Array.Accelerate (Vector)
import Control.Monad (sequence_)

main :: IO ()
main = defaultMain tests

tests :: TestTree
tests = testGroup "tests"
  [ tConstruction,
    tComposition,
    tElementWiseOperations
  ]

-- | Runs the given Accelerate computation on both interpreter and tensorflow backends and compares the results.
assertAcc :: (Arrays t, Eq t, Show t) => Acc t -> Assertion
assertAcc acc = run @TensorFlow acc @?= run @Interpreter acc

assertAcc2 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b) => (Acc a, Acc b) -> Assertion
assertAcc2 (a, b) = sequence_ 
  [ run @TensorFlow a @?= run @Interpreter a, 
    run @TensorFlow b @?= run @Interpreter b
  ]

assertAcc3 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c) => (Acc a, Acc b, Acc c) -> Assertion
assertAcc3 (a, b, c) = sequence_ 
  [ run @TensorFlow a @?= run @Interpreter a, 
    run @TensorFlow b @?= run @Interpreter b,
    run @TensorFlow c @?= run @Interpreter c
  ]

assertAcc4 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d) => (Acc a, Acc b, Acc c, Acc d) -> Assertion
assertAcc4 (a, b, c, d) = sequence_ 
  [ run @TensorFlow a @?= run @Interpreter a, 
    run @TensorFlow b @?= run @Interpreter b,
    run @TensorFlow c @?= run @Interpreter c,
    run @TensorFlow d @?= run @Interpreter d
  ]

assertAcc5 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e) => (Acc a, Acc b, Acc c, Acc d, Acc e) -> Assertion
assertAcc5 (a, b, c, d, e) = sequence_ 
  [ run @TensorFlow a @?= run @Interpreter a, 
    run @TensorFlow b @?= run @Interpreter b,
    run @TensorFlow c @?= run @Interpreter c,
    run @TensorFlow d @?= run @Interpreter d,
    run @TensorFlow e @?= run @Interpreter e
  ]

assertAcc6 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e, Arrays f, Eq f, Show f) => (Acc a, Acc b, Acc c, Acc d, Acc e, Acc f) -> Assertion
assertAcc6 (a, b, c, d, e, f) = sequence_ 
  [ run @TensorFlow a @?= run @Interpreter a, 
    run @TensorFlow b @?= run @Interpreter b,
    run @TensorFlow c @?= run @Interpreter c,
    run @TensorFlow d @?= run @Interpreter d,
    run @TensorFlow e @?= run @Interpreter e,
    run @TensorFlow f @?= run @Interpreter f
  ]

assertAcc7 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e, Arrays f, Eq f, Show f, Arrays g, Eq g, Show g) => (Acc a, Acc b, Acc c, Acc d, Acc e, Acc f, Acc g) -> Assertion
assertAcc7 (a, b, c, d, e, f, g) = sequence_ 
  [ run @TensorFlow a @?= run @Interpreter a, 
    run @TensorFlow b @?= run @Interpreter b,
    run @TensorFlow c @?= run @Interpreter c,
    run @TensorFlow d @?= run @Interpreter d,
    run @TensorFlow e @?= run @Interpreter e,
    run @TensorFlow f @?= run @Interpreter f,
    run @TensorFlow g @?= run @Interpreter g
  ]

assertAcc8 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e, Arrays f, Eq f, Show f, Arrays g, Eq g, Show g, Arrays h, Eq h, Show h) => (Acc a, Acc b, Acc c, Acc d, Acc e, Acc f, Acc g, Acc h) -> Assertion
assertAcc8 (a, b, c, d, e, f, g, h) = sequence_ 
  [ run @TensorFlow a @?= run @Interpreter a, 
    run @TensorFlow b @?= run @Interpreter b,
    run @TensorFlow c @?= run @Interpreter c,
    run @TensorFlow d @?= run @Interpreter d,
    run @TensorFlow e @?= run @Interpreter e,
    run @TensorFlow f @?= run @Interpreter f,
    run @TensorFlow g @?= run @Interpreter g,
    run @TensorFlow h @?= run @Interpreter h
  ]

assertAcc9 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e, Arrays f, Eq f, Show f, Arrays g, Eq g, Show g, Arrays h, Eq h, Show h, Arrays i, Eq i, Show i) => (Acc a, Acc b, Acc c, Acc d, Acc e, Acc f, Acc g, Acc h, Acc i) -> Assertion
assertAcc9 (a, b, c, d, e, f, g, h, i) = sequence_ 
  [ run @TensorFlow a @?= run @Interpreter a, 
    run @TensorFlow b @?= run @Interpreter b,
    run @TensorFlow c @?= run @Interpreter c,
    run @TensorFlow d @?= run @Interpreter d,
    run @TensorFlow e @?= run @Interpreter e,
    run @TensorFlow f @?= run @Interpreter f,
    run @TensorFlow g @?= run @Interpreter g,
    run @TensorFlow h @?= run @Interpreter h,
    run @TensorFlow i @?= run @Interpreter i
  ]

tConstruction :: TestTree
tConstruction = testGroup "Construction"
  [
    tIntroduction,
    tInitialisation,
    tEnumeration,
    tConcatenation,
    tExpansion
  ]
  where tIntroduction = testGroup "Introduction"
          [ tUse,
            tUnit
          ]
          where tUse = testGroup "use"
                  [ testCase "use vec" $ assertAcc $ use (fromList (Z:.10) [0..] :: Vector Int64),
                    testCase "use mat" $ assertAcc $ use (fromList (Z:.5:.10) [0..] :: Matrix Int64),
                    testCase "use tup" $ assertAcc $ use (fromList (Z:.10) [0..] :: Vector Int64, fromList (Z:.5:.10) [0..] :: Matrix Int64)
                  ]
                tUnit = testGroup "unit"
                  [ testCase "unit 1" $ assertAcc $ unit (constant 1 :: Exp Int64),
                    testCase "unit (1, 2)" $ assertAcc $ unit (constant (1, 2) :: Exp (Int64, Int64))
                  ]
        tInitialisation = testGroup "Initialisation"
          [ tGenerate,
            tFill
          ]
          where tGenerate = testGroup "Generate"
                  [ testCase "generate 1.2s" $ assertAcc (generate (I1 3) (const 1.2) :: Acc (Array DIM1 Float)),
                    testCase "generate [1..]" $ assertAcc (generate (I1 10) (\(I1 i) -> fromIntegral $ i + 1) :: Acc (Array DIM1 Int64))
                  ]
                tFill = testGroup "Fill"
                  [ testCase "fill 1.2s" $ assertAcc (fill (constant (Z :. 3)) 1.2 :: Acc (Array DIM1 Float))
                  ]
        tEnumeration = testGroup "Enumeration"
          [ tEnumFromN,
            tEnumFromStepN
          ]
          where tEnumFromN = testGroup "enumFromN"
                  [ testCase "x+1" $ assertAcc (enumFromN (constant (Z:.5:.10)) 0 :: Acc (Array DIM2 Int64))
                  ]
                tEnumFromStepN = testGroup "enumFromStepN"
                  [ testCase "x+ny" $ assertAcc (enumFromStepN (constant (Z:.5:.10)) 0 0.5 :: Acc (Array DIM2 Float))
                  ]
        tConcatenation = testGroup "Concatenation"
          [ tPlusPlus
            --tConcatOn -- TODO: it seems like _2 is not imported
          ]
          where tPlusPlus = testGroup "++"
                  [ testCase "++" $ assertAcc (use (fromList (Z:.5:.10) [0..]) ++ use (fromList (Z:.10:.3) [0..]) :: Acc (Array DIM2 Int64))
                  ]
                -- tConcatOn = let 
                --       m1 = fromList (Z:.5:.10) [0..] :: Matrix Int64
                --       m2 = fromList (Z:.10:.5) [0..] :: Matrix Int64
                --     in testGroup "concatOn"
                --   [ testCase "concatOn _1" $ assertAcc (concatOn _1 (use m1) (use m2)),
                --     testCase "concatOn _2" $ assertAcc (concatOn _2 (use m1) (use m2))
                --   ]
        tExpansion = testGroup "Expansion"
          [ tExpand
          ]
          where tExpand = let primes :: Exp Int -> Acc (Vector Int)
                              primes n = afst loop
                                where
                                  c0    = unit 2
                                  a0    = use $ fromList (Z:.0) []
                                  limit = truncate (sqrt (fromIntegral (n+1) :: Exp Float))
                                  loop  = awhile
                                            (\(T2 _   c) -> map (< n+1) c)
                                            (\(T2 old c) ->
                                              let c1 = the c
                                                  c2 = c1 < limit ? ( c1*c1, n+1 )
                                                  --
                                                  sieves =
                                                    let sz p    = (c2 - p) `quot` p
                                                        get p i = (2+i)*p
                                                    in
                                                    map (subtract c1) (expand sz get old)
                                                  --
                                                  new =
                                                    let m     = c2-c1
                                                        put i = let s = sieves ! i
                                                                in s >= 0 && s < m ? (Just_ (I1 s), Nothing_)
                                                    in
                                                    afst
                                                      $ filter (> 0)
                                                      $ permute const (enumFromN (I1 m) c1) put
                                                      $ fill (shape sieves) 0
                                              in
                                              T2 (old ++ new) (unit c2))
                                            (T2 a0 c0)
                  in testGroup "expand"
                  [ testCase "expand" $ assertAcc $ primes 100
                  ]

tComposition :: TestTree
tComposition = testGroup "Composition"
  [ tFlowControl,
    tControllingExecution
  ]
  where tFlowControl = testGroup "Flow Control"
          [ tIacond,
            tAcond,
            tAWhile,
            tIfThenElse
          ]
          where tIacond = testGroup "infix acond (?|)"
                  [ testCase "(?|) true" $ assertAcc (True_ ?| (use $ fromList (Z:.5:.10) [0..], use $ fromList (Z:.5:.10) [1..]) :: Acc (Array DIM2 Int64)),
                    testCase "(?|) false" $ assertAcc (False_ ?| (use $ fromList (Z:.5:.10) [0..], use $ fromList (Z:.5:.10) [1..]) :: Acc (Array DIM2 Int64))
                  ]
                tAcond = testGroup "acond"
                  [ testCase "acond true" $ assertAcc (acond True_ (use $ fromList (Z:.5:.10) [0..]) (use $ fromList (Z:.5:.10) [1..]) :: Acc (Array DIM2 Int64)),
                    testCase "acond false" $ assertAcc (acond False_ (use $ fromList (Z:.5:.10) [0..]) (use $ fromList (Z:.5:.10) [1..]) :: Acc (Array DIM2 Int64))
                  ]
                tAWhile = testGroup "awhile"
                  [ testCase "awhile" $ assertAcc (awhile (fold (&&) True_ . map (<= 10)) (map (+ 1)) (use $ fromList (Z:.10) [0..] :: Acc (Array DIM1 Int64)) :: Acc (Array DIM1 Int64))
                  ]
                tIfThenElse = testGroup "if then else"
                  [ testCase "if then else" $ assertAcc (ifThenElse (constant True) (use $ fromList (Z:.5:.10) [0..]) (use $ fromList (Z:.5:.10) [1..]) :: Acc (Array DIM2 Int64))
                  ]
        tControllingExecution = testGroup "Controlling Execution"
          [ tPipeline,
            tCompute
          ]
          where tPipeline = testGroup "pipeline (>->)"
                  [ testCase "pipeline (>->)" $ assertAcc ((>->) (map (+1)) (map (/ 2)) (use (fromList (Z:.5:.10) [1..])) :: Acc (Array DIM2 Float))
                  ]
                tCompute = let 
                  loop :: Exp Int -> Exp Int
                  loop ticks = let clockRate = 900000   -- kHz
                               in  while (\i -> i < clockRate * ticks) (+1) 0
                  in testGroup "compute"
                  [ testCase "compute" $ assertAcc $ zip3
                      (compute $ map loop (use $ fromList (Z:.1) [10]))
                      (compute $ map loop (use $ fromList (Z:.1) [10]))
                      (compute $ map loop (use $ fromList (Z:.1) [10]))
                  ]

tElementWiseOperations :: TestTree
tElementWiseOperations = testGroup "Element-wise operations"
  [ tIndexing,
    tMapping,
    tZipping,
    tUnzipping
  ]
  where tIndexing = testGroup "Indexing"
          [ tIndexed
          ]
          where tIndexed = testGroup "indexed"
                  [ testCase "indexed vec" $ assertAcc (indexed (use (fromList (Z:.5) [0..] :: Vector Float))),
                    testCase "indexed mat" $ assertAcc (indexed (use (fromList (Z:.3:.4) [0..] :: Matrix Float)))
                  ]
        tMapping = testGroup "Mapping"
          [ tMap,
            tIMap
          ]
          where tMap = testGroup "map"
                  [ testCase "map" $ assertAcc (map (+1) (use (fromList (Z:.5) [0..] :: Vector Float)))
                  ]
                tIMap = testGroup "imap"
                  [ testCase "imap" $ assertAcc (imap (\(I1 i) x -> x + fromIntegral i) (use (fromList (Z:.5) [0..] :: Vector Int64)))
                  ]
        tZipping = testGroup "Zipping"
          [ tZipWith,
            tZipWith3,
            tZipWith4,
            tZipWith5,
            tZipWith6,
            tZipWith7,
            tZipWith8,
            tZipWith9,
            tIZipWith,
            tIZipWith3,
            tIZipWith4,
            tIZipWith5,
            tIZipWith6,
            tIZipWith7,
            tIZipWith8,
            tIZipWith9,
            tZip,
            tZip3,
            tZip4,
            tZip5,
            tZip6,
            tZip7,
            tZip8,
            tZip9
          ]
          where vec = use (fromList (Z:.5) [1..] :: Vector Int64)
                tZipWith = testGroup "zipWith"
                  [ testCase "zipWith" $ assertAcc (zipWith (+) vec vec)
                  ]
                tZipWith3 = testGroup "zipWith3"
                  [ testCase "zipWith3" $ assertAcc (zipWith3 (\a b c -> a + b + c) vec vec vec)
                  ]
                tZipWith4 = testGroup "zipWith4"
                  [ testCase "zipWith4" $ assertAcc (zipWith4 (\a b c d -> a + b + c + d) vec vec vec vec)
                  ]
                tZipWith5 = testGroup "zipWith5"
                  [ testCase "zipWith5" $ assertAcc (zipWith5 (\a b c d e -> a + b + c + d + e) vec vec vec vec vec)
                  ]
                tZipWith6 = testGroup "zipWith6"
                  [ testCase "zipWith6" $ assertAcc (zipWith6 (\a b c d e f -> a + b + c + d + e + f) vec vec vec vec vec vec)
                  ]
                tZipWith7 = testGroup "zipWith7"
                  [ testCase "zipWith7" $ assertAcc (zipWith7 (\a b c d e f g-> a + b + c + d + e + f + g) vec vec vec vec vec vec vec)
                  ]
                tZipWith8 = testGroup "zipWith8"
                  [ testCase "zipWith8" $ assertAcc (zipWith8 (\a b c d e f g h -> a + b + c + d + e + f + g + h) vec vec vec vec vec vec vec vec)
                  ]  
                tZipWith9 = testGroup "zipWith9"
                  [ testCase "zipWith9" $ assertAcc (zipWith9 (\a b c d e f g h i -> a + b + c + d + e + f + g + h + i) vec vec vec vec vec vec vec vec vec)
                  ]
                tIZipWith = testGroup "izipWith"
                  [ testCase "izipWith" $ assertAcc (izipWith (\(I1 i) a b -> a + b + fromIntegral i) vec vec)
                  ]
                tIZipWith3 = testGroup "izipWith3"
                  [ testCase "izipWith3" $ assertAcc (izipWith3 (\(I1 i) a b c -> a + b + c + fromIntegral i) vec vec vec)
                  ]
                tIZipWith4 = testGroup "izipWith4"
                  [ testCase "izipWith4" $ assertAcc (izipWith4 (\(I1 i) a b c d -> a + b + c + d + fromIntegral i) vec vec vec vec)
                  ]
                tIZipWith5 = testGroup "izipWith5"
                  [ testCase "izipWith5" $ assertAcc (izipWith5 (\(I1 i) a b c d e -> a + b + c + d + e + fromIntegral i) vec vec vec vec vec)
                  ]
                tIZipWith6 = testGroup "izipWith6"
                  [ testCase "izipWith6" $ assertAcc (izipWith6 (\(I1 i) a b c d e f -> a + b + c + d + e + f + fromIntegral i) vec vec vec vec vec vec)
                  ]
                tIZipWith7 = testGroup "izipWith7"
                  [ testCase "izipWith7" $ assertAcc (izipWith7 (\(I1 i) a b c d e f g -> a + b + c + d + e + f + g + fromIntegral i) vec vec vec vec vec vec vec)
                  ]
                tIZipWith8 = testGroup "izipWith8"
                  [ testCase "izipWith8" $ assertAcc (izipWith8 (\(I1 i) a b c d e f g h -> a + b + c + d + e + f + g + h + fromIntegral i) vec vec vec vec vec vec vec vec)
                  ]
                tIZipWith9 = testGroup "izipWith9"
                  [ testCase "izipWith9" $ assertAcc (izipWith9 (\(I1 i) a b c d e f g h j -> a + b + c + d + e + f + g + h + j + fromIntegral i) vec vec vec vec vec vec vec vec vec)
                  ]
                tZip = testGroup "zip"
                  [ testCase "zip" $ assertAcc (zip vec vec)
                  ]
                tZip3 = testGroup "zip3"
                  [ testCase "zip3" $ assertAcc (zip3 vec vec vec)
                  ]
                tZip4 = testGroup "zip4"
                  [ testCase "zip4" $ assertAcc (zip4 vec vec vec vec)
                  ]
                tZip5 = testGroup "zip5"
                  [ testCase "zip5" $ assertAcc (zip5 vec vec vec vec vec)
                  ]
                tZip6 = testGroup "zip6"
                  [ testCase "zip6" $ assertAcc (zip6 vec vec vec vec vec vec)
                  ]
                tZip7 = testGroup "zip7"
                  [ testCase "zip7" $ assertAcc (zip7 vec vec vec vec vec vec vec)
                  ]
                tZip8 = testGroup "zip8"
                  [ testCase "zip8" $ assertAcc (zip8 vec vec vec vec vec vec vec vec)
                  ]
                tZip9 = testGroup "zip9"
                  [ testCase "zip9" $ assertAcc (zip9 vec vec vec vec vec vec vec vec vec)
                  ]                
        tUnzipping = testGroup "Unzipping"
          [
            tUnzip,
            tUnzip3,
            tUnzip4,
            tUnzip5,
            tUnzip6,
            tUnzip7,
            tUnzip8,
            tUnzip9
          ]
          where vec = use (fromList (Z:.5) [1..] :: Vector Int64)
                tUnzip = testGroup "unzip"
                  [ testCase "unzip" $ assertAcc2 (unzip $ zip vec vec :: (Acc (Vector Int64), Acc (Vector Int64))) 
                  ]
                tUnzip3 = testGroup "unzip3"
                  [ testCase "unzip3" $ assertAcc3 (unzip3 $ zip3 vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64))) 
                  ]
                tUnzip4 = testGroup "unzip4"
                  [ testCase "unzip4" $ assertAcc4 (unzip4 $ zip4 vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64))) 
                  ]
                tUnzip5 = testGroup "unzip5"
                  [ testCase "unzip5" $ assertAcc5 (unzip5 $ zip5 vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64))) 
                  ]
                tUnzip6 = testGroup "unzip6"
                  [ testCase "unzip6" $ assertAcc6 (unzip6 $ zip6 vec vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64))) 
                  ]
                tUnzip7 = testGroup "unzip7"
                  [ testCase "unzip7" $ assertAcc7 (unzip7 $ zip7 vec vec vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64))) 
                  ]
                tUnzip8 = testGroup "unzip7"
                  [ testCase "unzip8" $ assertAcc8 (unzip8 $ zip8 vec vec vec vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64))) 
                  ]
                tUnzip9 = testGroup "unzip9"
                  [ testCase "unzip9" $ assertAcc9 (unzip9 $ zip9 vec vec vec vec vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64))) 
                  ]