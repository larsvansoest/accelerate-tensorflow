{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RebindableSyntax #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
import Data.Array.Accelerate hiding (Eq, Vector)
import Prelude hiding (uncurry, curry, (^^), (^), lcm, gcd, (||), not, iterate, scaleFloat, isNaN, isInfinite, isDenormalized, isNegativeZero, atan2, isIEEE, significand, exponent, encodeFloat, decodeFloat, floatRange, floatDigits, floatRadix, properFraction, floor, ceiling, round, toRational, compare, min, (/=), (==), scanr1, scanr, scanl1, scanl, Ord, maximum, minimum, product, or, and, any, all, max, odd, even, reverse, Num, drop, take, tail, init, replicate, unzip3, unzip, zip, zipWith3, zip3, (<=), (>), filter, (&&), (>=), subtract, (<), truncate, (++), fromIntegral, map, (+))
import Data.Accelerate.TensorFlow.Execute
import Data.Array.Accelerate.Interpreter

import Test.Tasty
import Test.Tasty.HUnit
import Data.Array.Accelerate (Vector)
import Control.Monad (sequence_)
import Data.Array.Accelerate.Data.Ratio (Ratio, (%))
import Data.Accelerate.TensorFlow.Operation
import Data.Accelerate.TensorFlow.Desugar
import Data.Accelerate.TensorFlow.Kernel
import Data.Array.Accelerate.Pretty.Schedule
import Data.Array.Accelerate.AST.Schedule.Sequential hiding (Exp)
import Data.Array.Accelerate.Pretty.Schedule.Sequential

type Stencil5x1 a = (Stencil3 a, Stencil5 a, Stencil3 a)
type Stencil1x5 a = (Stencil3 a, Stencil3 a, Stencil3 a, Stencil3 a, Stencil3 a)

-- TODO: Add tests for: test for different sizes for zipwith (ongelijke matrices maakt gebruik van minimum dimensions)

-- main :: IO ()
-- main = let x = use (fromList (Z:.5:.10) [0..])
--            z = shape x
--            y = map (Data.Array.Accelerate.fromIndex (shape x)) x
--        in putStrLn $ show $ run @TensorFlow y

main :: IO ()
main = defaultMain tests

tests :: TestTree
tests = testGroup "tests"
  [ tAccelerateArrayLanguage,
    tAccelerateExpressionLanguage
  ]

-- | Runs the given Accelerate computation on both interpreter and tensorflow backends and compares the results.
assertAcc :: (Arrays t, Eq t, Show t) => Acc t -> Assertion
assertAcc acc = run @TensorFlow acc @?= run @Interpreter acc

assertAcc2 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b) => (Acc a, Acc b) -> Assertion
assertAcc2 (a, b) = sequence_
  [ assertAcc a,
    assertAcc b
  ]

assertAcc3 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c) => (Acc a, Acc b, Acc c) -> Assertion
assertAcc3 (a, b, c) = sequence_
  [ assertAcc a,
    assertAcc b,
    assertAcc c
  ]

assertAcc4 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d) => (Acc a, Acc b, Acc c, Acc d) -> Assertion
assertAcc4 (a, b, c, d) = sequence_
  [ assertAcc a,
    assertAcc b,
    assertAcc c,
    assertAcc d
  ]

assertAcc5 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e) => (Acc a, Acc b, Acc c, Acc d, Acc e) -> Assertion
assertAcc5 (a, b, c, d, e) = sequence_
  [ assertAcc a,
    assertAcc b,
    assertAcc c,
    assertAcc d,
    assertAcc e
  ]

assertAcc6 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e, Arrays f, Eq f, Show f) => (Acc a, Acc b, Acc c, Acc d, Acc e, Acc f) -> Assertion
assertAcc6 (a, b, c, d, e, f) = sequence_
  [ assertAcc a,
    assertAcc b,
    assertAcc c,
    assertAcc d,
    assertAcc e,
    assertAcc f
  ]

assertAcc7 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e, Arrays f, Eq f, Show f, Arrays g, Eq g, Show g) => (Acc a, Acc b, Acc c, Acc d, Acc e, Acc f, Acc g) -> Assertion
assertAcc7 (a, b, c, d, e, f, g) = sequence_
  [ assertAcc a,
    assertAcc b,
    assertAcc c,
    assertAcc d,
    assertAcc e,
    assertAcc f,
    assertAcc g
  ]

assertAcc8 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e, Arrays f, Eq f, Show f, Arrays g, Eq g, Show g, Arrays h, Eq h, Show h) => (Acc a, Acc b, Acc c, Acc d, Acc e, Acc f, Acc g, Acc h) -> Assertion
assertAcc8 (a, b, c, d, e, f, g, h) = sequence_
  [ assertAcc a,
    assertAcc b,
    assertAcc c,
    assertAcc d,
    assertAcc e,
    assertAcc f,
    assertAcc g,
    assertAcc h
  ]

assertAcc9 :: (Arrays a, Eq a, Show a, Arrays b, Eq b, Show b, Arrays c, Eq c, Show c, Arrays d, Eq d, Show d, Arrays e, Eq e, Show e, Arrays f, Eq f, Show f, Arrays g, Eq g, Show g, Arrays h, Eq h, Show h, Arrays i, Eq i, Show i) => (Acc a, Acc b, Acc c, Acc d, Acc e, Acc f, Acc g, Acc h, Acc i) -> Assertion
assertAcc9 (a, b, c, d, e, f, g, h, i) = sequence_
  [ assertAcc a,
    assertAcc b,
    assertAcc c,
    assertAcc d,
    assertAcc e,
    assertAcc f,
    assertAcc g,
    assertAcc h,
    assertAcc i
  ]

tAccelerateArrayLanguage :: TestTree
tAccelerateArrayLanguage = testGroup "The Accelerate Array Language"
  [ tConstruction,
    tComposition,
    tElementWiseOperations,
    tModifyingArrays,
    tFolding,
    tScans,
    tStencils
  ]
  where tConstruction :: TestTree
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
                          [ testCase "(?|) true" $ assertAcc (True_ ?| (use $ fromList (Z:.5:.10) [0..], use $ fromList (Z:.5:.10) [5..]) :: Acc (Array DIM2 Int64)),
                            testCase "(?|) false" $ assertAcc (False_ ?| (use $ fromList (Z:.5:.10) [0..], use $ fromList (Z:.5:.10) [5..]) :: Acc (Array DIM2 Int64))
                          ]
                        tAcond = testGroup "acond"
                          [ testCase "acond true" $ assertAcc (acond True_ (use $ fromList (Z:.5:.10) [0..]) (use $ fromList (Z:.5:.10) [5..]) :: Acc (Array DIM2 Int64)),
                            testCase "acond false" $ assertAcc (acond False_ (use $ fromList (Z:.5:.10) [0..]) (use $ fromList (Z:.5:.10) [5..]) :: Acc (Array DIM2 Int64))
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
                          [ testCase "zipWith" $ assertAcc (Data.Array.Accelerate.zipWith (+) vec vec)
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

        tModifyingArrays :: TestTree
        tModifyingArrays = testGroup "Modifying Arrays"
          [ tShapeManipulation,
            tReplication,
            tExtractingSubArrays,
            tPermutations,
            tFiltering
          ]
          where tShapeManipulation = testGroup "Shape manipulation"
                  [ tReshape,
                    tFlatten
                  ]
                  where tReshape = testGroup "reshape"
                          [ testCase "reshape" $ assertAcc (reshape (constant (Z:.5:.5)) (use (fromList (Z:.25) [1..] :: Vector Int64)))
                          ]
                        tFlatten = testGroup "flatten"
                          [ testCase "flatten" $ assertAcc (flatten (use (fromList (Z:.5:.5) [1..] :: Matrix Int64)))
                          ]
                tReplication = testGroup "Replication"
                  [ tReplicate
                  ]
                  where tReplicate = let
                            vec = fromList (Z:.10) [0..] :: Vector Int
                            rep0 :: (Shape sh, Elt e) => Exp Int -> Acc (Array sh e) -> Acc (Array (sh :. Int) e)
                            rep0 n a = replicate (lift (Any :. n)) a
                            rep1 :: (Shape sh, Elt e) => Exp Int -> Acc (Array (sh :. Int) e) -> Acc (Array (sh :. Int :. Int) e)
                            rep1 n a = replicate (lift (Any :. n :. All)) a
                          in testGroup "replicate"
                          [ testCase "replicate 2d" $ assertAcc (replicate (constant (Z :. (4 :: Int) :. All)) (use vec)),
                            testCase "replicate 2d columns" $ assertAcc (replicate (lift (Z :. All :. (4::Int))) (use vec)),
                            testCase "replicate 2x1d, 3x 3d" $ assertAcc (replicate (constant (Z :. (2::Int) :. All :. (3::Int))) (use vec)),
                            testCase "rep0 1d" $ assertAcc $ rep0 10 (unit 42 :: Acc (Scalar Int)),
                            testCase "rep0 2d" $ assertAcc $ rep0 5 (use vec),
                            testCase "rep1" $ assertAcc $ rep1 5 (use vec)
                          ]
                tExtractingSubArrays = testGroup "Extracting subarrays"
                  [ tSlice,
                    tInit,
                    tTail,
                    tTake,
                    tDrop,
                    tSlit,
                    tInitOn,
                    tTailOn,
                    tTakeOn,
                    tDropOn,
                    tSlitOn
                  ]
                  where tSlice = let
                            mat = fromList (Z:.5:.10) [0..] :: Matrix Int
                            sl0 :: (Shape sh, Elt e) => Acc (Array (sh:.Int) e) -> Exp Int -> Acc (Array sh e)
                            sl0 a n = slice a (lift (Any :. n))
                            vec = fromList (Z:.10) [0..] :: Vector Int
                            sl1 :: (Shape sh, Elt e) => Acc (Array (sh:.Int:.Int) e) -> Exp Int -> Acc (Array (sh:.Int) e)
                            sl1 a n = slice a (lift (Any :. n :. All))
                            cube = fromList (Z:.3:.4:.5) [0..] :: Array DIM3 Int
                          in testGroup "slice"
                          [ testCase "slice mat 1d" $ assertAcc (slice (use mat) (constant (Z :. (2 :: Int) :. All))),
                            testCase "slice mat 0d" $ assertAcc (slice (use mat) (constant (Z :. 4 :. 2 :: DIM2))),
                            testCase "sl0 vec" $ assertAcc $ sl0 (use vec) 4,
                            testCase "sl0 mat" $ assertAcc $ sl0 (use mat) 4,
                            testCase "sl1 mat" $ assertAcc $ sl1 (use mat) 4,
                            testCase "sl1 cube" $ assertAcc $ sl1 (use cube) 2
                          ]
                        tInit = testGroup "init"
                          [ testCase "init mat" $ assertAcc (init (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "init vec" $ assertAcc (init (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tTail = testGroup "tail"
                          [
                            testCase "tail mat" $ assertAcc (tail (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "tail vec" $ assertAcc (tail (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tTake = testGroup "take"
                          [ testCase "take mat" $ assertAcc (take (constant 3) (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "take vec" $ assertAcc (take (constant 3) (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tDrop = testGroup "drop"
                          [ testCase "drop mat" $ assertAcc (drop (constant 3) (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "drop vec" $ assertAcc (drop (constant 3) (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tSlit = testGroup "slit"
                          [ testCase "slit mat" $ assertAcc (slit (constant 1) (constant 3) (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "slit vec" $ assertAcc (slit (constant 1) (constant 3) (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tInitOn = testGroup "initOn"
                          [ -- testCase "initOn mat" $ assertAcc (initOn (Z :. (1 :: Int) :. All)) (use (fromList (Z:.5:.10) [0..] :: Matrix Int64)),
                            -- testCase "initOn vec" $ assertAcc (initOn (Z :. (1 :: Int)) (use (fromList (Z:.10) [0..] :: Vector Int64))) -- Todo: fix lens
                          ]
                        tTailOn = testGroup "tailOn"
                          [ -- Todo: fix lens
                          ]
                        tTakeOn =  testGroup "tTakeOn"
                          [ -- Todo: fix lens
                          ]
                        tDropOn = testGroup "tDropOn"
                          [ -- Todo: fix lens
                          ]
                        tSlitOn = testGroup "tSlitOn"
                          [ -- Todo: fix lens
                          ]
                tPermutations = testGroup "Permutations"
                  [ tForwardPermutation,
                    tBackwardPermutation,
                    tSpecialisedPermutations
                  ]
                  where tForwardPermutation = testGroup "Forward Permutation (scatter)"
                          [ tPermute,
                            tScatter
                          ]
                          where tPermute = let
                                    histogram :: Acc (Vector Int) -> Acc (Vector Int)
                                    histogram xs =
                                      let zeros = fill (constant (Z:.10)) 0
                                          ones  = fill (shape xs)         1
                                      in
                                      permute (+) zeros (\ix -> Just_ (I1 (xs!ix))) ones
                                    identity :: Num a => Exp Int -> Acc (Matrix a)
                                    identity n =
                                      let zeros = fill (I2 n n) 0
                                          ones  = fill (I1 n)   1
                                      in
                                      permute const zeros (\(I1 i) -> Just_ (I2 i i)) ones
                                  in testGroup "permute"
                                  [ testCase "histogram" $ assertAcc $ histogram (use (fromList (Z :. 20) [0,0,1,2,1,1,2,4,8,3,4,9,8,3,2,5,5,3,1,2] :: Vector Int)),
                                    testCase "identity" $ assertAcc (identity 5 :: Acc (Matrix Int64))
                                  ]
                                tScatter = let
                                    to    = fromList (Z :. 6) [1,3,7,2,5,8] :: Vector Int
                                    input = fromList (Z :. 7) [1,9,6,4,4,2,5] :: Vector Int
                                  in testGroup "scatter"
                                  [ testCase "scatter" $ assertAcc $ scatter (use to) (fill (constant (Z:.10)) 0) (use input)
                                  ]
                        tBackwardPermutation = testGroup "Backward Permutation (gather)"
                          [ tBackpermute,
                            tGather
                          ]
                          where tBackpermute = let
                                    swap :: Exp DIM2 -> Exp DIM2
                                    swap = lift1 f
                                      where
                                        f :: Z :. Exp Int :. Exp Int -> Z :. Exp Int :. Exp Int
                                        f (Z:.y:.x) = Z :. x :. y
                                    mat = fromList (Z:.5:.10) [0..] :: Matrix Int
                                    mat' = use mat
                                  in testGroup "backpermute"
                                  [ testCase "backpermute swap" $ assertAcc $ backpermute (swap (shape mat')) swap mat'
                                  ]
                                tGather = let
                                    from  = fromList (Z :. 6) [1,3,7,2,5,8] :: Vector Int
                                    input = fromList (Z :. 7) [1,9,6,4,4,2,5] :: Vector Int
                                  in testGroup "gather"
                                  [ testCase "gather" $ assertAcc $ gather (use from) (use input)
                                  ]
                        tSpecialisedPermutations = testGroup "Specialised Permutations"
                          [
                            tReverse,
                            tTranspose,
                            tReverseOn,
                            tTransposeOn
                          ]
                          where tReverse = testGroup "reverse"
                                  [ testCase "reverse vec" $ assertAcc (reverse (use (fromList (Z :. 10) [0..] :: Vector Int64)))
                                  ]
                                tTranspose = testGroup "transpose"
                                  [ testCase "transpose mat" $ assertAcc (transpose (use (fromList (Z :. 5 :. 10) [0..] :: Matrix Int64)))
                                  ]
                                tReverseOn = testGroup "reverseOn"
                                  [ -- TODO: fix lens
                                  ]
                                tTransposeOn = testGroup "transposeOn"
                                  [ -- TODO: fix lens
                                  ]
                tFiltering = testGroup "Filtering"
                  [ tFilter,
                    tCompact
                  ]
                  where vec = fromList (Z :. 10) [1..10] :: Vector Int64
                        mat = fromList (Z :. 4 :. 10) [1,2,3,4,5,6,7,8,9,10,1,1,1,1,1,2,2,2,2,2,2,4,6,8,10,12,14,16,18,20,1,3,5,7,9,11,13,15,17,19] :: Matrix Int
                        tFilter = testGroup "filter"
                          [ testCase "filter even" $ assertAcc (filter even (use vec)),
                            testCase "filter odd" $ assertAcc (filter odd (use mat))
                          ]
                        tCompact = testGroup "compact"
                          [ testCase "compact even" $ assertAcc (compact (map even $ use vec) $ use vec),
                            testCase "compact odd" $ assertAcc (compact (map odd $ use mat) $ use mat)
                          ]

        tFolding :: TestTree
        tFolding = testGroup "Folding"
          [ tFold,
            tFold1,
            tFoldAll,
            tFold1All,
            tSegmentedReductions,
            tSpecialisedReductions
          ]
          where maximumSegmentSum
                      :: forall sh e. (Shape sh, Num e, Ord e)
                      => Acc (Array (sh :. Int) e)
                      -> Acc (Array sh e)
                maximumSegmentSum
                  = map (\(T4 x _ _ _) -> x)
                  . fold1 f
                  . map g
                  where
                    f :: (Num a, Ord a) => Exp (a,a,a,a) -> Exp (a,a,a,a) -> Exp (a,a,a,a)
                    f x y =
                      let T4 mssx misx mcsx tsx = x
                          T4 mssy misy mcsy tsy = y
                      in
                      T4 (mssx `max` (mssy `max` (mcsx+misy)))
                        (misx `max` (tsx+misy))
                        (mcsy `max` (mcsx+tsy))
                        (tsx+tsy)
                    --
                    g :: (Num a, Ord a) => Exp a -> Exp (a,a,a,a)
                    g x = let y = max x 0
                          in T4 y y y x
                tFold = testGroup "fold"
                  [ testCase "fold + 42" $ assertAcc (fold (+) 42 (use (fromList (Z:.5:.10) [0..] :: Matrix Int))),
                    testCase "fold maximumSegmentSum" $ assertAcc (maximumSegmentSum (use (fromList (Z:.10) [-2,1,-3,4,-1,2,1,-5,4,0] :: Vector Int)))
                  ]
                tFold1 = testGroup "fold1"
                  [ testCase "fold1 + 42" $ assertAcc (fold1 (+) (use (fromList (Z:.5:.10) [0..] :: Matrix Int))),
                    testCase "fold1 maximumSegmentSum" $ assertAcc (fold1 max (use (fromList (Z:.10) [-2,1,-3,4,-1,2,1,-5,4,0] :: Vector Int)))
                  ]
                tFoldAll = testGroup "foldAll"
                  [ testCase "foldAll + 42 vec" $ assertAcc (foldAll (+) 42 (use (fromList (Z:.10) [0..] :: Vector Float))),
                    testCase "foldAll + 0 mat" $ assertAcc (foldAll (+) 0 (use (fromList (Z:.5:.10) [0..] :: Matrix Float)))
                  ]
                tFold1All = testGroup "fold1All"
                  [ testCase "fold1All + 42 vec" $ assertAcc (fold1All (+) (use (fromList (Z:.10) [0..] :: Vector Float))),
                    testCase "fold1All + 0 mat" $ assertAcc (fold1All (+) (use (fromList (Z:.5:.10) [0..] :: Matrix Float)))
                  ]
                tSegmentedReductions = testGroup "Segmented Reductions"
                  [ tFoldSeg,
                    tFold1Seg
                  ]
                  where tFoldSeg = testGroup "foldSeg"
                          [ testCase "foldSeg vec" $ assertAcc (foldSeg (+) 0 (use (fromList (Z :. 10) [0..] :: Vector Int64)) (use (fromList (Z :. 10) [0..] :: Vector Int64))),
                            testCase "foldSeg mat" $ assertAcc (foldSeg (+) 0 (use (fromList (Z :. 5 :. 10) [0..] :: Matrix Int64)) (use (fromList (Z :. 5) [0..] :: Vector Int64)))
                          ]
                        tFold1Seg = testGroup "fold1Seg"
                          [ testCase "fold1Seg vec" $ assertAcc (fold1Seg (+) (use (fromList (Z :. 10) [0..] :: Vector Int64)) (use (fromList (Z :. 10) [0..] :: Vector Int64))),
                            testCase "fold1Seg mat" $ assertAcc (fold1Seg (+) (use (fromList (Z :. 5 :. 10) [0..] :: Matrix Int64)) (use (fromList (Z :. 5) [0..] :: Vector Int64)))
                          ]
                tSpecialisedReductions = testGroup "Specialised Reductions"
                  [ tAll,
                    tAny,
                    tAnd,
                    tOr,
                    tSum,
                    tProduct,
                    tMinimum,
                    tMaximum
                  ]
                  where vec = fromList (Z :. 10) [0..] :: Vector Int64
                        mat = fromList (Z :. 4 :. 10) [1,2,3,4,5,6,7,8,9,10,1,1,1,1,1,2,2,2,2,2,2,4,6,8,10,12,14,16,18,20,1,3,5,7,9,11,13,15,17,19] :: Matrix Int64
                        allTrue = fromList (Z :. 10) [True, True, True, True, True, True, True, True, True, True] :: Vector Bool
                        someFalse = fromList (Z :. 10) [True, True, False, True, True, True, True, True, True, True] :: Vector Bool
                        allFalse = fromList (Z :. 10) [False, False, False, False, False, False, False, False, False, False] :: Vector Bool
                        tAll = testGroup "all"
                          [ testCase "all vec" $ assertAcc (all even (use vec)),
                            testCase "all mat" $ assertAcc (all even (use mat))
                          ]
                        tAny = testGroup "any"
                          [ testCase "any vec" $ assertAcc (any even (use vec)),
                            testCase "any mat" $ assertAcc (any even (use mat))
                          ]
                        tAnd = testGroup "and"
                          [ testCase "and vec all True" $ assertAcc (and (use allTrue)),
                            testCase "and vec some False" $ assertAcc (and (use someFalse)),
                            testCase "and vec all False" $ assertAcc (and (use allFalse))
                          ]
                        tOr = testGroup "or"
                          [ testCase "or vec all True" $ assertAcc (or (use allTrue)),
                            testCase "or vec some False" $ assertAcc (or (use someFalse)),
                            testCase "or vec all False" $ assertAcc (or (use allFalse))
                          ]
                        tSum = testGroup "sum"
                          [ testCase "sum vec" $ assertAcc (Data.Array.Accelerate.sum (use vec)),
                            testCase "sum mat" $ assertAcc (Data.Array.Accelerate.sum (use mat))
                          ]
                        tProduct = testGroup "product"
                          [ testCase "product vec" $ assertAcc (product (use (fromList (Z :. 10) [1..] :: Vector Int64))),
                            testCase "product mat" $ assertAcc (product (use mat))
                          ]
                        tMinimum = testGroup "minimum"
                          [ testCase "minimum vec" $ assertAcc (minimum (use vec)),
                            testCase "minimum mat" $ assertAcc (minimum (use mat))
                          ]
                        tMaximum = testGroup "maximum"
                          [ testCase "maximum vec" $ assertAcc (maximum (use vec)),
                            testCase "maximum mat" $ assertAcc (maximum (use mat))
                          ]

        tScans :: TestTree
        tScans = testGroup "Scans (prefix sums)"
          [ tScanl,
            tScanl1,
            tScanl',
            tScanr,
            tScanr1,
            tScanr',
            tPrescanl,
            tPostscanl,
            tPrescanr,
            tPostscanr,
            tSegmentedScans
          ]
          where vec = fromList (Z :. 10) [0..] :: Vector Int64
                mat = fromList (Z :. 4 :. 10) [1,2,3,4,5,6,7,8,9,10,1,1,1,1,1,2,2,2,2,2,2,4,6,8,10,12,14,16,18,20,1,3,5,7,9,11,13,15,17,19] :: Matrix Int64
                tScanl = testGroup "scanl"
                  [ testCase "scanl vec" $ assertAcc (scanl (+) 10 (use vec)),
                    testCase "scanl mat" $ assertAcc (scanl (+) 42 (use mat))
                  ]
                tScanl1 = testGroup "scanl1"
                  [ testCase "scanl1 vec" $ assertAcc (scanl1 (+) (use vec)),
                    testCase "scanl1 mat" $ assertAcc (scanl1 (+) (use mat))
                  ]
                tScanl' = testGroup "scanl'"
                  [ testCase "scanl' vec" $ assertAcc (scanl' (+) 10 (use vec)),
                    testCase "scanl' mat" $ assertAcc (scanl' (+) 42 (use mat))
                  ]
                tScanr = testGroup "scanr"
                  [ testCase "scanr vec" $ assertAcc (scanr (+) 10 (use vec)),
                    testCase "scanr mat" $ assertAcc (scanr (+) 42 (use mat))
                  ]
                tScanr1 = testGroup "scanr1"
                  [ testCase "scanr1 vec" $ assertAcc (scanr1 (+) (use vec)),
                    testCase "scanr1 mat" $ assertAcc (scanr1 (+) (use mat))
                  ]
                tScanr' = testGroup "scanr'"
                  [ testCase "scanr' vec" $ assertAcc (scanr' (+) 10 (use vec)),
                    testCase "scanr' mat" $ assertAcc (scanr' (+) 42 (use mat))
                  ]
                tPrescanl = testGroup "prescanl"
                  [ testCase "prescanl vec" $ assertAcc (prescanl (+) 10 (use vec)),
                    testCase "prescanl mat" $ assertAcc (prescanl (+) 42 (use mat))
                  ]
                tPostscanl = testGroup "postscanl"
                  [ testCase "postscanl vec" $ assertAcc (postscanl (+) 10 (use vec)),
                    testCase "postscanl mat" $ assertAcc (postscanl (+) 42 (use mat))
                  ]
                tPrescanr = testGroup "prescanr"
                  [ testCase "prescanr vec" $ assertAcc (prescanr (+) 10 (use vec)),
                    testCase "prescanr mat" $ assertAcc (prescanr (+) 42 (use mat))
                  ]
                tPostscanr = testGroup "postscanr"
                  [ testCase "postscanr vec" $ assertAcc (postscanr (+) 10 (use vec)),
                    testCase "postscanr mat" $ assertAcc (postscanr (+) 42 (use mat))
                  ]
                tSegmentedScans = testGroup "segmented scans"
                  [ tScanlSeg,
                    tScanl1Seg,
                    tScanl'Seg,
                    tPrescanlSeg,
                    tPostscanlSeg,
                    tScanrSeg,
                    tScanr1Seg,
                    tScanr'Seg,
                    tPrescanrSeg,
                    tPostscanrSeg
                  ]
                  where seg = fromList (Z:.4) [1,4,0,3] :: Segments Int
                        mat = fromList (Z:.5:.10) [0..] :: Matrix Int
                        tScanlSeg = testGroup "scanlSeg"
                          [ testCase "scanlSeg mat" $ assertAcc (scanlSeg (+) 0 (use mat) (use seg))
                          ]
                        tScanl1Seg = testGroup "scanl1Seg"
                          [ testCase "scanl1Seg mat" $ assertAcc (scanl1Seg (+) (use mat) (use seg))
                          ]
                        tScanl'Seg = testGroup "scanl'Seg"
                          [ testCase "scanl'Seg mat" $ assertAcc (scanl'Seg (+) 0 (use mat) (use seg))
                          ]
                        tPrescanlSeg = testGroup "prescanlSeg"
                          [ testCase "prescanlSeg mat" $ assertAcc (prescanlSeg (+) 0 (use mat) (use seg))
                          ]
                        tPostscanlSeg = testGroup "postscanlSeg"
                          [ testCase "postscanlSeg mat" $ assertAcc (postscanlSeg (+) 0 (use mat) (use seg))
                          ]
                        tScanrSeg = testGroup "scanrSeg"
                          [ testCase "scanrSeg mat" $ assertAcc (scanrSeg (+) 0 (use mat) (use seg))
                          ]
                        tScanr1Seg = testGroup "scanr1Seg"
                          [ testCase "scanr1Seg mat" $ assertAcc (scanr1Seg (+) (use mat) (use seg))
                          ]
                        tScanr'Seg = testGroup "scanr'Seg"
                          [ testCase "scanr'Seg mat" $ assertAcc (scanr'Seg (+) 0 (use mat) (use seg))
                          ]
                        tPrescanrSeg = testGroup "prescanrSeg"
                          [ testCase "prescanrSeg mat" $ assertAcc (prescanrSeg (+) 0 (use mat) (use seg))
                          ]
                        tPostscanrSeg = testGroup "postscanrSeg"
                          [ testCase "postscanrSeg mat" $ assertAcc (postscanrSeg (+) 0 (use mat) (use seg))
                          ]

        tStencils :: TestTree
        tStencils = testGroup "Stencils"
          [ tStencil,
            tStencil2
          ]
          where mat = fromList (Z :. 4 :. 10) [1,2,3,4,5,6,7,8,9,10,1,1,1,1,1,2,2,2,2,2,2,4,6,8,10,12,14,16,18,20,1,3,5,7,9,11,13,15,17,19] :: Matrix Float
                convolve5x1 :: Num a => [Exp a] -> Stencil5x1 a -> Exp a
                convolve5x1 kernel (_, (a,b,c,d,e), _)
                  = Prelude.sum $ Prelude.zipWith (*) kernel [a,b,c,d,e]
                convolve1x5 :: Num a => [Exp a] -> Stencil1x5 a -> Exp a
                convolve1x5 kernel ((_,a,_), (_,b,_), (_,c,_), (_,d,_), (_,e,_))
                  = Prelude.sum $ Prelude.zipWith (*) kernel [a,b,c,d,e]
                gaussian = [0.06136,0.24477,0.38774,0.24477,0.06136] :: [Exp Float]
                blur :: Acc (Matrix Float) -> Acc (Matrix Float)
                blur = stencil (convolve5x1 gaussian) clamp
                     . stencil (convolve1x5 gaussian) clamp
                tStencil = testGroup "stencil"
                  [ testCase "stencil blur" $ assertAcc $ blur $ use mat
                  ]
                tStencil2 = testGroup "stencil2"
                  [ -- TODO: test stencil2
                  ]

tAccelerateExpressionLanguage :: TestTree
tAccelerateExpressionLanguage = testGroup "The Accelerate Expression Language"
  [ tBasicTypes,
    tNumTypes,
    -- TODO: numeric conversion classes
    tScalarOperations,
    -- TODO: tForeignFunctionInterface,
    tPlainArrays
  ]
  where tBasicTypes :: TestTree
        tBasicTypes = testGroup "Basic type classes"
          [ tEq,
            tOrd
          ]
          where vec1 = fromList (Z:.1) [1] :: Vector Int64
                tEq = testGroup "Eq"
                  [ tEquals,
                    tNotEquals
                  ]
                  where tEquals = testGroup "equals (==)"
                          [ testCase "eqInt True" $ assertAcc (map (== 1) (use vec1)),
                            testCase "eqInt False" $ assertAcc (map (== 0) (use vec1))
                          ]
                        tNotEquals = testGroup "not equals (/=)"
                          [ testCase "neqInt True" $ assertAcc (map (/= 1) (use vec1)),
                            testCase "neqInt False" $ assertAcc (map (/= 0) (use vec1))
                          ]
                tOrd = testGroup "Ord"
                  [ tLessThan,
                    tLessThanEquals,
                    tGreaterThan,
                    tGreaterThanEquals,
                    tMin,
                    tMax,
                    tCompare
                  ]
                  where tLessThan = testGroup "less than (<)"
                          [ testCase "ltInt True" $ assertAcc (map (< 2) (use vec1)),
                            testCase "ltInt False" $ assertAcc (map (< 1) (use vec1))
                          ]
                        tLessThanEquals = testGroup "less than equals (<=)"
                          [ testCase "lteInt True" $ assertAcc (map (<= 1) (use vec1)),
                            testCase "lteInt False" $ assertAcc (map (<= 0) (use vec1))
                          ]
                        tGreaterThan = testGroup "greater than (>)"
                          [ testCase "gtInt True" $ assertAcc (map (> 0) (use vec1)),
                            testCase "gtInt False" $ assertAcc (map (> 1) (use vec1))
                          ]
                        tGreaterThanEquals = testGroup "greater than equals (>=)"
                          [ testCase "gteInt True" $ assertAcc (map (>= 1) (use vec1)),
                            testCase "gteInt False" $ assertAcc (map (>= 2) (use vec1))
                          ]
                        tMin = testGroup "min"
                          [ testCase "minInt l" $ assertAcc (map (min 0) (use vec1)),
                            testCase "minInt r" $ assertAcc (map (min 2) (use vec1))
                          ]
                        tMax = testGroup "max"
                          [ testCase "maxInt l" $ assertAcc (map (max 2) (use vec1)),
                            testCase "maxInt r" $ assertAcc (map (max 0) (use vec1))
                          ]
                        tCompare = testGroup "compare"
                          [ testCase "compareInt l" $ assertAcc (map (compare 0) (use vec1)),
                            testCase "compareInt r" $ assertAcc (map (compare 2) (use vec1))
                          ]

        tNumTypes :: TestTree
        tNumTypes = testGroup "Numeric type classes"
          [ tNum,
            tRational,
            tFractional,
            tFloating,
            tRealFrac,
            tRealFloat
          ]
          where vec1 = fromList (Z:.10) [1..] :: Vector Int64
                vec1' = fromList (Z:.10) [1..] :: Vector Float
                tNum = testGroup "Num"
                  [
                    tPlus,
                    tMinus,
                    tTimes,
                    tNegate,
                    tAbs,
                    tSignum,
                    tQuot,
                    tRem,
                    tDiv,
                    tMod,
                    tQuotRem,
                    tDivMod
                  ]
                  where tPlus = testGroup "plus (+)"
                          [ testCase "plus" $ assertAcc (map (+ 1) (use vec1))
                          ]
                        tMinus = testGroup "minus (-)"
                          [ testCase "minus" $ assertAcc (map (\x -> x - 1) (use vec1) :: Acc (Vector Int64))
                          ]
                        tTimes = testGroup "times (*)"
                          [ testCase "times" $ assertAcc (map (* 2) (use vec1))
                          ]
                        tNegate = testGroup "negate"
                          [ testCase "negate" $ assertAcc (map negate (use vec1))
                          ]
                        tAbs = testGroup "abs"
                          [ testCase "abs" $ assertAcc (map abs (use vec1))
                          ]
                        tSignum = testGroup "signum"
                          [ testCase "signum" $ assertAcc (map signum (use vec1))
                          ]
                        tQuot = testGroup "quot"
                          [ testCase "quot" $ assertAcc (map (quot 2) (use vec1))
                          ]
                        tRem = testGroup "rem"
                          [ testCase "rem" $ assertAcc (map (rem 2) (use vec1))
                          ]
                        tDiv = testGroup "div"
                          [ testCase "div" $ assertAcc (map (div 2) (use vec1))
                          ]
                        tMod = testGroup "mod"
                          [ testCase "mod" $ assertAcc (map (mod 2) (use vec1))
                          ]
                        tQuotRem = testGroup "quotRem"
                          [ testCase "quotRem" $ assertAcc (map (Prelude.fst . quotRem 2) (use vec1)),
                            testCase "quotRem" $ assertAcc (map (Prelude.snd . quotRem 2) (use vec1))
                          ]
                        tDivMod = testGroup "divMod"
                          [ testCase "divMod" $ assertAcc (map (Prelude.fst . divMod 2) (use vec1)),
                            testCase "divMod" $ assertAcc (map (Prelude.snd . divMod 2) (use vec1))
                          ]

                tRational = testGroup "Rational"
                  [ tToRational,
                    tFromRational
                  ]
                  where tToRational = testGroup "toRational"
                          [ testCase "toRational" $ assertAcc (map toRational (use vec1) :: Acc (Vector (Ratio Int64)))
                          ]
                        tFromRational = testGroup "fromRational"
                          [ --testCase "fromRational" $ assertAcc (map fromRational (use vecRatio1)),
                            --testCase "fromRational" $ assertAcc (map fromRational (use vecRatio1))
                          ]

                tFractional = testGroup "Fractional"
                  [ tDivide,
                    tRecip
                    -- tFromRational
                  ]
                  where tDivide = testGroup "divide (/)"
                          [ testCase "divide" $ assertAcc (map (/ (2 :: Exp Float)) (use vec1'))
                          ]
                        tRecip = testGroup "recip"
                          [ testCase "recip" $ assertAcc (map recip (use vec1'))
                          ]

                tFloating = testGroup "Floating"
                  [ tPi,
                    tSin,
                    tCos,
                    tTan,
                    tAsin,
                    tAcos,
                    tAtan,
                    tSinh,
                    tCosh,
                    tTanh,
                    tAsinh,
                    tAcosh,
                    tAtanh,
                    tExp,
                    tSqrt,
                    tLog,
                    tFPow,
                    tLogBase
                  ]
                  where vec1 = fromList (Z:.1) [1] :: Vector Int64
                        tPi = testGroup "pi"
                          [ --testCase "pi" $ assertAcc (map pi (use vec1') :: Acc (Vector Float)), -- TODO: fix
                          ]
                        tSin = testGroup "sin"
                          [ testCase "sin" $ assertAcc (map sin (use vec1'))
                          ]
                        tCos = testGroup "cos"
                          [ testCase "cos" $ assertAcc (map cos (use vec1'))
                          ]
                        tTan = testGroup "tan"
                          [ testCase "tan" $ assertAcc (map tan (use vec1'))
                          ]
                        tAsin = testGroup "asin"
                          [ testCase "asin" $ assertAcc (map asin (use vec1'))
                          ]
                        tAcos = testGroup "acos"
                          [ testCase "acos" $ assertAcc (map acos (use vec1'))
                          ]
                        tAtan = testGroup "atan"
                          [ testCase "atan" $ assertAcc (map atan (use vec1'))
                          ]
                        tSinh = testGroup "sinh"
                          [ testCase "sinh" $ assertAcc (map sinh (use vec1'))
                          ]
                        tCosh = testGroup "cosh"
                          [ testCase "cosh" $ assertAcc (map cosh (use vec1'))
                          ]
                        tTanh = testGroup "tanh"
                          [ testCase "tanh" $ assertAcc (map tanh (use vec1'))
                          ]
                        tAsinh = testGroup "asinh"
                          [ testCase "asinh" $ assertAcc (map asinh (use vec1'))
                          ]
                        tAcosh = testGroup "acosh"
                          [ testCase "acosh" $ assertAcc (map acosh (use vec1'))
                          ]
                        tAtanh = testGroup "atanh"
                          [ testCase "atanh" $ assertAcc (map atanh (use vec1'))
                          ]
                        tExp = testGroup "exp"
                          [ testCase "exp" $ assertAcc (map exp (use vec1'))
                          ]
                        tSqrt = testGroup "sqrt"
                          [ testCase "sqrt" $ assertAcc (map sqrt (use vec1'))
                          ]
                        tLog = testGroup "log"
                          [ testCase "log" $ assertAcc (map log (use vec1'))
                          ]
                        tFPow = testGroup "**"
                          [ testCase "**" $ assertAcc (map (** (2 :: Exp Float)) (use vec1'))
                          ]
                        tLogBase = testGroup "logBase"
                          [ testCase "logBase" $ assertAcc (map (logBase (2 :: Exp Float)) (use vec1'))
                          ]


                tRealFrac = testGroup "RealFrac"
                  [ tProperFraction,
                    tTruncate,
                    tRound,
                    tCeiling,
                    tFloor,
                    tDiv',
                    tMod',
                    tDivMod'
                  ]
                  where tProperFraction = testGroup "properFraction"
                          [ -- testCase "properFraction" $ assertAcc (map (Prelude.snd . properFraction) (use vec1) :: Acc (Vector Int64))
                          ]
                        tTruncate = testGroup "truncate"
                          [ testCase "truncate" $ assertAcc (map truncate (use vec1') :: Acc (Vector Int64))
                          ]
                        tRound = testGroup "round"
                          [ testCase "truncate" $ assertAcc (map round (use vec1') :: Acc (Vector Int64))
                          ]
                        tCeiling = testGroup "ceiling"
                          [ testCase "ceiling" $ assertAcc (map ceiling (use vec1') :: Acc (Vector Int64))
                          ]
                        tFloor = testGroup "floor"
                          [ testCase "floor" $ assertAcc (map floor (use vec1') :: Acc (Vector Int64))
                          ]
                        tDiv' = testGroup "div"
                          [ testCase "div" $ assertAcc (map (div (2 :: Exp Int64)) (use vec1) :: Acc (Vector Int64))
                          ]
                        tMod' = testGroup "mod"
                          [ testCase "mod" $ assertAcc (map (mod (2 :: Exp Int64)) (use vec1) :: Acc (Vector Int64))
                          ]
                        tDivMod' = testGroup "divMod"
                          [ --testCase "divMod" $ assertAcc (map (Data.Array.Accelerate.fst . divMod' (2 :: Exp Int64)) (use vec1) :: Acc (Vector (Int64, Int64)))
                          ]

                tRealFloat = testGroup "Real"
                  [ tFloatRadix,
                    tFloatDigits,
                    tFloatRange,
                    tDecodeFloat,
                    tEncodeFloat,
                    tExponent,
                    tSignificand,
                    tScaleFloat,
                    tIsNaN,
                    tIsInfinite,
                    tIsDenormalized,
                    tIsNegativeZero,
                    tIsIEEE,
                    tAtan2
                  ]
                  where tFloatRadix = testGroup "floatRadix"
                          [ testCase "floatRadix" $ assertAcc (map floatRadix (use vec1'))
                          ]
                        tFloatDigits = testGroup "floatDigits"
                          [ testCase "floatDigits" $ assertAcc (map floatDigits (use vec1'))
                          ]
                        tFloatRange = testGroup "floatRange"
                          [ testCase "floatRange" $ assertAcc (map (Prelude.fst . floatRange) (use vec1'))
                          ]
                        tDecodeFloat = testGroup "decodeFloat"
                          [ testCase "decode" $ assertAcc (map (Prelude.fst . decodeFloat) (use vec1'))
                          ]
                        tEncodeFloat = testGroup "encodeFloat"
                          [ testCase "encode" $ assertAcc (map (encodeFloat 1) (use (fromList (Z:.10) [1..10] :: Vector Int)) :: Acc (Vector Float))
                          ]
                        tExponent = testGroup "exponent"
                          [ testCase "exponent" $ assertAcc (map exponent (use vec1'))
                          ]
                        tSignificand = testGroup "significand"
                          [ testCase "significand" $ assertAcc (map significand (use vec1'))
                          ]
                        tScaleFloat = testGroup "scale"
                          [ testCase "scale" $ assertAcc (map (scaleFloat 1) (use vec1'))
                          ]
                        tIsNaN = testGroup "isNaN"
                          [ testCase "isNaN" $ assertAcc (map isNaN (use vec1'))
                          ]
                        tIsInfinite = testGroup "isInfinite"
                          [ testCase "isInfinite" $ assertAcc (map isInfinite (use vec1'))
                          ]
                        tIsDenormalized = testGroup "isDenormalized"
                          [ testCase "isDenormalized" $ assertAcc (map isDenormalized (use vec1'))
                          ]
                        tIsNegativeZero = testGroup "isNegativeZero"
                          [ testCase "isNegativeZero" $ assertAcc (map isNegativeZero (use vec1'))
                          ]
                        tIsIEEE = testGroup "isIEEE"
                          [ testCase "isIEEE" $ assertAcc (map isIEEE (use vec1'))
                          ]
                        tAtan2 = testGroup "atan2"
                          [ testCase "atan2" $ assertAcc (map (atan2 1) (use vec1'))
                          ]

        tScalarOperations :: TestTree
        tScalarOperations = testGroup "Scalar Operations"
          [ tIntroduction,
            tTuples,
            tFlowControl,
            tScalarReduction,
            tLogicalOperations,
            tNumericOperations,
            tShapeManipulation,
            tConversions
          ]
          where tIntroduction = testGroup "Introduction"
                  [ tConstant
                  ]
                  where vec = fromList (Z:.5) [1..5] :: Vector Int64
                        tConstant = testGroup "constant"
                          [ testCase "constant" $ assertAcc (map (const (constant (1 :: Int))) (use vec))
                          ]
                tTuples = testGroup "Tuples"
                  [ tFst,
                    tAfst,
                    tSnd,
                    tAsnd,
                    tCurry,
                    tUncurry
                  ]
                  where ones = fromList (Z:.5) [1..] :: Vector Int64
                        zeroes = fromList (Z:.5) [0..] :: Vector Int64
                        tFst = testGroup "fst"
                          [ testCase "fst" $ assertAcc (map Data.Array.Accelerate.fst $ zip (use ones) (use zeroes))
                          ]
                        tAfst = testGroup "afst"
                          [ --testCase "afst" $ assertAcc (map afst $ zip (use ones) (use zeroes))
                          ]
                        tSnd = testGroup "snd"
                          [ testCase "snd" $ assertAcc (map Data.Array.Accelerate.snd $ zip (use ones) (use zeroes))
                          ]
                        tAsnd = testGroup "asnd"
                          [ --testCase "asnd" $ assertAcc (map asnd $ zip (use ones) (use zeroes))
                          ]
                        tCurry = testGroup "curry"
                          [ --testCase "curry" $ assertAcc (map (curry (+)) $ zip (use ones) (use zeroes))
                          ]
                        tUncurry = testGroup "uncurry"
                          [ testCase "uncurry" $ assertAcc (map (uncurry (+)) $ zip (use ones) (use zeroes))
                          ]

                tFlowControl = testGroup "Flow Control"
                  [ tQuestionMark,
                    tMatch,
                    tCond,
                    tWhile,
                    tIterate
                  ]
                  where tQuestionMark = testGroup "questionMark"
                          [ --testCase "questionMark" $ assertAcc (Data.Array.Accelerate.zipWith (?) (use (fromList (Z:.3) [True, False, True] :: Vector Bool)) (use (fromList (Z:.3) [(1,2), (2, 3), (3, 4)] :: Vector (Int, Int))))
                          ]
                        tMatch = testGroup "match"
                          [ --
                          ]
                        tCond = testGroup "cond"
                          [ testCase "cond" $ assertAcc (zipWith3 cond (use (fromList (Z:.3) [True, False, True] :: Vector Bool)) (use (fromList (Z:.3) [1, 2, 3] :: Vector Int)) (use (fromList (Z:.3) [4, 5, 6] :: Vector Int)))
                          ]
                        tWhile = testGroup "while"
                          [ --testCase "while" 
                          ]
                        tIterate = testGroup "iterate"
                          [ -- testCase "iterate" 
                            -- testCase "iterate" 
                          ]
                tScalarReduction = testGroup "Scalar Reduction"
                  [
                    tSfoldl
                  ]
                  where vec1 = fromList (Z:.5:.10) [1..] :: Matrix Int
                        tSfoldl = testGroup "sfoldl"
                          [ -- testCase "sfoldl" $ assertAcc (sfoldl (\(I2 x y -> x + y) 0 (use vec1)))
                          ]
                tLogicalOperations = testGroup "Logical Operations"
                  [
                    tAnd,
                    tOr,
                    tNot
                  ]
                  where vecTrue = fromList (Z:.5:.10) (repeat True) :: Matrix Bool
                        vecFalse = fromList (Z:.5:.10) (repeat False) :: Matrix Bool
                        tAnd = testGroup "and (&&)"
                          [ testCase "and True True" $ assertAcc (Data.Array.Accelerate.zipWith (&&) (use vecTrue) (use vecFalse)),
                            testCase "and True False" $ assertAcc (Data.Array.Accelerate.zipWith (&&) (use vecTrue) (use vecFalse)),
                            testCase "and False True" $ assertAcc (Data.Array.Accelerate.zipWith (&&) (use vecFalse) (use vecTrue)),
                            testCase "and False False" $ assertAcc (Data.Array.Accelerate.zipWith (&&) (use vecFalse) (use vecFalse))
                          ]
                        tOr = testGroup "or (||)"
                          [ testCase "or True True" $ assertAcc (Data.Array.Accelerate.zipWith (||) (use vecTrue) (use vecFalse)),
                            testCase "or True False" $ assertAcc (Data.Array.Accelerate.zipWith (||) (use vecTrue) (use vecFalse)),
                            testCase "or False True" $ assertAcc (Data.Array.Accelerate.zipWith (||) (use vecFalse) (use vecTrue)),
                            testCase "or False False" $ assertAcc (Data.Array.Accelerate.zipWith (||) (use vecFalse) (use vecFalse))
                          ]
                        tNot = testGroup "not"
                          [ testCase "not True" $ assertAcc (map not (use vecTrue)),
                            testCase "not False" $ assertAcc (map not (use vecFalse))
                          ]
                tNumericOperations = testGroup "Numeric Operations"
                  [
                    tSubtract,
                    tEven,
                    tOdd,
                    tGcd,
                    tLcm,
                    tHat,
                    tHatHat
                  ]
                  where vec = fromList (Z:.5) [1..5] :: Vector Int64
                        vec' = fromList (Z:.5) [1..5] :: Vector Float
                        tSubtract = testGroup "subtract"
                          [ testCase "subtract" $ assertAcc (map (subtract 1) (use vec))
                          ]
                        tEven = testGroup "even"
                          [ testCase "even" $ assertAcc (map even (use vec))
                          ]
                        tOdd = testGroup "odd"
                          [ testCase "odd" $ assertAcc (map odd (use vec))
                          ]
                        tGcd = testGroup "gcd"
                          [ testCase "gcd" $ assertAcc (map (gcd 2) (use vec))
                          ]
                        tLcm = testGroup "lcm"
                          [ testCase "lcm" $ assertAcc (map (lcm 2) (use vec))
                          ]
                        tHat = testGroup "^"
                          [ testCase "^" $ assertAcc (map (^ (2 :: Exp Int64)) (use vec'))
                          ]
                        tHatHat = testGroup "^^"
                          [ testCase "^^" $ assertAcc (map (^^ (2 :: Exp Int64)) (use vec'))
                          ]
                tShapeManipulation = testGroup "Shape Manipulation"
                  [ tIndex0,
                    tIndex1,
                    tUnindex1,
                    tIndex2,
                    tUnindex2,
                    tIndex3,
                    tUnindex3,
                    tIndexHead,
                    tIndexTail,
                    tToIndex,
                    tFromIndex,
                    tIntersect
                  ]
                  where tIndex0 = testGroup "index0"
                          [ testCase "index0" $ assertAcc (map (const index0) (use (fromList (Z:.3) [1, 2, 3] :: Vector Int)))
                          ]
                        tIndex1 = testGroup "index1"
                          [ --testCase "index1" $ 
                          ]
                        tUnindex1 = testGroup "unindex1"
                          [ --testCase "unindex1" $
                          ]
                        tIndex2 = testGroup "index2"
                          [ --testCase "index2" $ 
                          ]
                        tUnindex2 = testGroup "unindex2"
                          [ --testCase "unindex2" $ 
                          ]
                        tIndex3 = testGroup "index3"
                          [ --testCase "index3" $ 
                          ]
                        tUnindex3 = testGroup "unindex3"
                          [ --testCase "unindex3" $
                          ]
                        tIndexHead = testGroup "indexHead"
                          [ --testCase "indexHead" $
                          ]
                        tIndexTail = testGroup "indexTail"
                          [ --testCase "indexTail" $
                          ]
                        tToIndex = testGroup "toIndex"
                          [ --testCase "toIndex" $ 
                          ]
                        tFromIndex = testGroup "fromIndex"
                          [ --testCase "fromIndex" $
                          ]
                        tIntersect = testGroup "intersect"
                          [ --testCase "intersect" $
                          ]
                tConversions = testGroup "Conversions"
                  [ tOrd,
                    tChr,
                    tBoolToInt,
                    tBitcast
                  ]
                  where tOrd = testGroup "ord"
                          [ testCase "ord" $ assertAcc (map ord (use (fromList (Z:.3) ['a', 'b', 'c'] :: Vector Char)))
                          ]
                        tChr = testGroup "chr"
                          [ testCase "chr" $ assertAcc (map chr (use (fromList (Z:.3) [97, 98, 99] :: Vector Int)))
                          ]
                        tBoolToInt = testGroup "boolTo"
                          [ testCase "boolTo" $ assertAcc (map boolToInt (use (fromList (Z:.3) [True, False, True] :: Vector Bool)))
                          ]
                        tBitcast = testGroup "bitcast"
                          [ -- testCase "bitcast" $ assertAcc (map bitcast (use (fromList (Z:.3) [1, 2, 3] :: Vector Int)))
                          ]

        tPlainArrays :: TestTree
        tPlainArrays = testGroup "Plain Arrays"
          [ tOperations,
            tGettingDataIn
          ]
          where tOperations = testGroup "Operations"
                  [ tArrayRank,
                    tArrayShape,
                    tArraySize,
                    tArrayReshape,
                    tIndexArray,
                    tLinearIndexArray
                  ]
                  where tArrayRank = testGroup "arrayRank" -- TODO: how do I construct a plain array?
                          [ --testCase "arrayRank" $ assertAcc (use (arrayRank (constant (Z:.1:.2) )))
                          ]
                        tArrayShape = testGroup "arrayShape"
                          [ --testCase "arrayShape" $ assertAcc (use (arrayShape (constant (Z:.1:.2))))
                          ]
                        tArraySize = testGroup "arraySize"
                          [ --testCase "arraySize" $ assertAcc (use (arraySize (constant (Z:.1:.2))))
                          ]
                        tArrayReshape = testGroup "arrayReshape"
                          [ --testCase "arrayReshape" $ assertAcc (use (arrayReshape (constant (Z:.1:.2)) (constant (Z:.1:.2))))
                          ]
                        tIndexArray = testGroup "indexArray"
                          [ --testCase "indexArray" $ assertAcc (use (indexArray (constant (Z:.1:.2)) (constant (Z:.1:.2))))
                          ]
                        tLinearIndexArray = testGroup "linearIndexArray"
                          [ --testCase "linearIndexArray" $ assertAcc (use (linearIndexArray (constant (Z:.1:.2)) 1))
                          ]
                tGettingDataIn = testGroup "Getting Data In"
                  [ tFunctions,
                    tLists
                  ]
                  where tFunctions = testGroup "Functions"
                          [ tFromFunction,
                            tFromFunctionM
                          ]
                          where tFromFunction = testGroup "fromFunction"
                                  [ -- testCase "fromFunction" 
                                  ]
                                tFromFunctionM = testGroup "fromFunctionM"
                                  [ --testCase "fromFunctionM" $ assertAcc (use (fromFunctionM Identity (constant (Z:.1:.2)) (\((I2 x y) -> return (constant (x+y)))))) -- TODO: how to test i
                                  ]
                        tLists = testGroup "Lists"
                          [ tFromList,
                            tToList
                          ]
                          where tFromList = testGroup "fromList"
                                  [ testCase "fromList" $ assertAcc (use (fromList (Z:.1:.2) [1, 2] :: Array DIM2 Int))
                                  ]
                                tToList = testGroup "toList"
                                  [ -- testCase "toList" $ assertAcc (toList (use (fromList (Z:.1:.2) [1, 2] :: Array DIM2 Int)))
                                  ]

