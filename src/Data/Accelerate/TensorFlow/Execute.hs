
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}



{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use record patterns" #-}
{-# LANGUAGE TypeApplications #-}




module Data.Accelerate.TensorFlow.Execute where
import Data.Array.Accelerate.AST.Schedule.Sequential
import Data.Accelerate.TensorFlow.Kernel
import Data.Array.Accelerate hiding (map, sum, (++), fromIntegral, size, Vector, Exp)
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Text.Prettyprint.Doc (viaShow, align, group, vcat, vsep, hsep)
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.AST.Kernel
import Data.Array.Accelerate.Interpreter hiding (executeKernelFun, executeKernelFun, Right, Left)

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import Data.Vector (Vector)
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Data.IORef
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.AST.Environment hiding (push)
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Analysis.Match
import Control.Monad.IO.Class
import Unsafe.Coerce
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Trafo.Operation.Substitution
import qualified Data.Vector.Storable as S
import Control.Concurrent.MVar
import Data.Accelerate.TensorFlow.Type
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.Interpreter hiding (executeKernelFun)
import Data.Array.Accelerate.Interpreter (evalExpM)
import qualified Data.Array.Accelerate.Pretty as Pretty
import Data.Array.Accelerate.Pretty.Operation (prettyBuffer)
import Data.Array.Accelerate.Pretty.Exp (Adoc)
import Data.Array.Accelerate.Representation.Elt (showElt)


-- detect copy implementeren voor het versimpelen voor programma's

type TensorEnv = Env TensorValue

data TensorV a where
  TBuild :: TF.Tensor TF.Build a -> TensorV a
  TVector :: S.Vector a -> TensorV a
  TNil :: TensorV a

buildTensor :: (S.Storable a, TF.TensorDataType S.Vector a) => TF.Shape -> TensorV a -> TF.Tensor TF.Build a
buildTensor _ TNil           = error "can not build TNil"
buildTensor _ (TBuild build) = build
buildTensor sh (TVector vec) = TF.constant sh $ S.toList vec

data TensorValue a where
  TScalar :: TypeR a -> a -> TensorValue a
  TTensor :: (TF.TensorType a, Show a, S.Storable a, TF.TensorDataType S.Vector a) => ScalarType a -> TF.Shape -> IORef (TensorV a) -> TensorValue (Buffer a)

fromTTensor :: (TF.TensorType a, Show a, S.Storable a, TF.TensorDataType S.Vector a) => TensorValue (Buffer a) -> (ScalarType a, TF.Shape, IORef (TensorV a))
fromTTensor (TScalar _ _)       = error "wrong input for fromTTensor" 
fromTTensor (TTensor st sh ref) = (st, sh, ref)

type TensorValues = TupR TensorValue

type family TensorValuesIOFun t where
  TensorValuesIOFun (a -> b) = TensorValues a -> TensorValuesIOFun b
  TensorValuesIOFun t        = IO t

executeSequentialSchedule :: TensorEnv env -> SequentialSchedule TensorKernel env t -> TensorValuesIOFun t
executeSequentialSchedule env (SequentialLam lhs sched) = \values -> executeSequentialSchedule (push env (lhs, values)) sched
executeSequentialSchedule env (SequentialBody sched) = \_ -> do
  x <- TF.runSession $ executeSequentialSchedule' env sched
  runTensorValues x
  str <- prettyVectors x
  putStrLn $ Pretty.renderForTerminal str
  -- met de MVar moet ik nog iets: type class Execute implementeren

executeSequentialSchedule' :: TensorEnv env -> SeqSchedule TensorKernel env t -> TF.Session (TensorValues t)
executeSequentialSchedule' env (Exec _ fun args) = executeKernelFun env fun args

executeSequentialSchedule' env (Return vars) = return $ mapTupR (\(Var _ idx) -> prj' idx env) vars

executeSequentialSchedule' env (Compute expr) = do
  value <- evalExpM expr $ EvalArrayInstr $ evalArrayInstr env
  return $ TupRsingle (TScalar (expType expr) value)

executeSequentialSchedule' env (Alet lhs _ sched sched') = do
  rhs <- executeSequentialSchedule' env sched
  let env' = push env (lhs, rhs)
  executeSequentialSchedule' env' sched'

executeSequentialSchedule' env (Alloc shR st vars)
  | VectorTypeDict <- vectorTypeDict st
  = do
  sh <- liftIO $ getShape env shR vars
  ref <- liftIO $ newIORef TNil
  return $ TupRsingle $ TTensor st sh ref

executeSequentialSchedule' _ (Use st n buffer)
  | TensorTypeDict <- tensorTypeDict st
  , VectorTypeDict <- vectorTypeDict st
  = do
  let sh = TF.Shape [fromIntegral n]
  let build = TF.constant sh $ bufferToList st n buffer
  ref <- liftIO $ newIORef $ TBuild build
  return $ TupRsingle $ TTensor st sh ref

executeSequentialSchedule' _ (Unit var) = undefined

executeSequentialSchedule' _ (Acond var ss ss') = undefined

executeSequentialSchedule' _ (Awhile tr ssf ssf' tr') = undefined

evalArrayInstr :: TensorEnv env -> ArrayInstr env (s -> t) -> s -> TF.SessionT IO t -- write case matches
evalArrayInstr env (Index (Var _ idx)) s     = case prj' idx env of
  TScalar _ _ -> error "impossible"
  TTensor _ _ ref -> do
    tensorV <- liftIO $ readIORef ref
    case tensorV of
      TBuild _ -> undefined
      TVector vec -> return $ vec S.! s
      TNil -> error "impossible"
evalArrayInstr env (Parameter (Var _ idx)) _ = case prj' idx env of
  TScalar _ e -> return e
  TTensor {} -> error "impossible"

executeKernelFun :: TensorEnv env -> KernelFun TensorKernel args -> SArgs env args -> TF.Session (TensorValues ())
executeKernelFun = executeKernelFun' Empty

executeKernelFun' :: TensorEnv env' -> TensorEnv env -> OpenKernelFun TensorKernel env' args -> SArgs env args -> TF.Session (TensorValues ())
executeKernelFun' env' env (KernelFunLam (KernelArgRscalar _) fun) ((SArgScalar (Var _ idx)) :>: args)     = executeKernelFun' (env' `Push` prj' idx env) env fun args
executeKernelFun' env' env (KernelFunLam (KernelArgRbuffer _ _) fun) ((SArgBuffer _ (Var _ idx)) :>: args) = executeKernelFun' (env' `Push` prj' idx env) env fun args
executeKernelFun' env' _ (KernelFunBody kernel) ArgsNil                                                    = executeKernel env' kernel

executeKernel :: TensorEnv env -> TensorKernel env -> TF.Session (TensorValues ())
executeKernel env (TensorConstant _ t _ s (Var _ idx))
  | TensorTypeDict <- tensorTypeDict t
  , VectorTypeDict <- vectorTypeDict t
  = do
  let (_, TF.Shape sh, ref) = fromTTensor $ prj' idx env
  liftIO $ writeIORef ref $ TBuild $ TF.fill (TF.vector sh) (TF.scalar s)
  return TupRunit

-- executeKernel env (TensorPrimFun shR (PrimAdd nt) shVars (TupRpair (TupRsingle (Var _ inIdx1)) (TupRsingle (Var _ inIdx2))) (TupRsingle (Var _ outIdx)))
--   | TensorTypeDict <- tensorTypeDict' nt
--   , VectorTypeDict <- vectorTypeDict' nt
--   = do
--   let (_, sh1, inRef1) = fromTTensor $ prj' inIdx1 env
--   let (_, sh2, inRef2) = fromTTensor $ prj' inIdx2 env
--   inValue1 <- liftIO $ readIORef inRef1
--   inValue2 <- liftIO $ readIORef inRef2
--   let inTensor1 = buildTensor sh1 inValue1
--   let inTensor2 = buildTensor sh2 inValue2

--   let (_, _, outRef)   = fromTTensor $ prj' outIdx env
--   liftIO $ writeIORef outRef $ TBuild $ TF.add inTensor1 inTensor2
--   return TupRunit

executeKernel env (TensorPrimFun shR fun shVars inVars outVars) =
  return TupRunit

executeKernel env (TensorId _ st _ (Var _ inIdx) (Var _ outIdx)) 
  | TensorTypeDict <- tensorTypeDict st
  , VectorTypeDict <- vectorTypeDict st
  = do
  let (_, sh, inRef) = fromTTensor $ prj' inIdx env
  let (_, _, outRef) = fromTTensor $ prj' outIdx env
  inValue <- liftIO $ readIORef inRef
  liftIO $ writeIORef outRef $ TBuild $ buildTensor sh inValue
  return TupRunit

getShape :: TensorEnv env -> ShapeR sh -> ExpVars env sh -> IO TF.Shape
getShape env shR vars = do
  sh <- liftIO $ getShape' env shR vars
  return $ toTFShape shR sh
  where
    getShape' :: TensorEnv env -> ShapeR sh -> ExpVars env sh -> IO sh
    getShape' _ ShapeRz _                                                      = return ()
    getShape' env' (ShapeRsnoc shR') (TupRpair vars1 (TupRsingle (Var _ idx))) = do
      sh' <- getShape' env' shR' vars1
      return (sh', case prj' idx env' of { TScalar _ sh -> sh })
    getShape' _ _ _ = error "impossible"

push :: TensorEnv env -> (LeftHandSide s t env env', TensorValues t) -> TensorEnv env'
push env (LeftHandSideWildcard _, _)            = env
push env (LeftHandSideSingle _  , TupRsingle a) = env `Push` a
push env (LeftHandSidePair l1 l2, TupRpair a b) = push env (l1, a) `push` (l2, b)
push _ _                                        = error "Tuple mismatch"

type TensorBuffers = TupR TensorBuffer
data TensorBuffer t where
  TensorBuffer :: ScalarType t -> Int -> Buffer t -> TensorBuffer t

toTFShape :: ShapeR sh -> sh -> TF.Shape
toTFShape shR sh = TF.Shape $ fromIntegral <$> shapeToList shR sh

fromBuffer :: (TF.TensorType t, S.Storable t, TF.TensorDataType S.Vector t) =>
  ShapeR sh ->
  ScalarType t ->
  sh ->
  Buffer t ->
  TF.Session (S.Vector t)
fromBuffer shR t sh buffer = do
  let shape = toTFShape shR sh
  tensorData <- toTensorData t shape (size shR sh) buffer
  TF.runSession $ do
    x <- TF.placeholder shape
    let feeds = [ TF.feed x tensorData ]
    TF.runWithFeeds feeds $ TF.identity x

toTensorData :: (TF.TensorType t, S.Storable t, TF.TensorDataType S.Vector t) =>
  ScalarType t ->
  TF.Shape ->
  Int ->
  Buffer t ->
  TF.Session (TF.TensorData t)
toTensorData t sh n buffer = do
  let vec = S.fromList (bufferToList t n buffer)
  return $ TF.encodeTensorData sh vec

-- prettyBuffers :: TensorBuffers t -> String
-- prettyBuffers TupRunit        = "()"
-- prettyBuffers (TupRsingle (TensorBuffer st n buffer))  = Pretty.renderForTerminal $ prettyBuffer st n buffer
-- prettyBuffers (TupRpair t t') = "(" ++ prettyBuffers t ++ "," ++ prettyBuffers t' ++ ")"

prettyVectors :: TensorValues t -> IO Adoc
prettyVectors TupRunit                     = return $ viaShow "()"
prettyVectors (TupRsingle (TScalar st s))  = return $ viaShow $ fromString $ showElt st s
prettyVectors (TupRsingle (TTensor st _ ref)) = do
  value <- readIORef ref
  case value of
    TBuild _    -> error "tensorvalue refers to build, please run tensor values first"
    TNil        -> return $ viaShow "TNil"
    TVector vec -> return $ prettyVector st vec
prettyVectors (TupRpair v1 v2) = do
  doc1 <- prettyVectors v1
  doc2 <- prettyVectors v2
  return $ hsep [ "(", doc1, ",", doc2, ")"]

prettyVector :: ScalarType a -> S.Vector a -> Adoc
prettyVector st vec 
 | VectorTypeDict <- vectorTypeDict st
 = align $ group $ "( " <> vcat (mapTail (", " <>) $ map (fromString . showElt (TupRsingle st)) $ S.toList vec) <> " )"

mapTail :: (a -> a) -> [a] -> [a]
mapTail f (x:xs) = x : map f xs
mapTail _ []     = []

-- toBuffers :: TensorValues t -> IO (TensorBuffers t)
-- toBuffers TupRunit         = return TupRunit
-- toBuffers (TupRsingle v)   = do buffer <- toBuffer _ v
--                                 return $ TupRsingle buffer
-- toBuffers (TupRpair v1 v2) = do buffers1 <- toBuffers v1
--                                 buffers2 <- toBuffers v2
--                                 return (TupRpair buffers1 buffers2)

-- toBuffer :: ScalarType a -> TensorValue a -> IO (TensorBuffer a)
-- toBuffer st (TScalar _ _) = error "impossible"
-- toBuffer st (TTensor _ ref)
--   = do
--     value <- readIORef ref
--     case value of
--       TBuild _    -> error "tensorvalue refers to build, please run tensor values first"
--       TNil        -> error "cannot convert TNil to buffer"
--       TVector vec -> do
--         let n = S.length vec
--         mutableBuffer <- newBuffer st n
--         writeVectorToBuffer st vec mutableBuffer
--         return $ TensorBuffer st n $ unsafeFreezeBuffer mutableBuffer

-- writeVectorToBuffer :: ScalarType a -> S.Vector a -> MutableBuffer a -> IO ()
-- writeVectorToBuffer st vec buffer = S.iforM_ vec (writeBuffer st buffer)

runTensorValues :: TensorValues t -> IO ()
runTensorValues TupRunit         = return ()
runTensorValues (TupRsingle v)   = runTensorValue v
runTensorValues (TupRpair v1 v2) = do runTensorValues v1
                                      runTensorValues v2

runTensorValue :: TensorValue t -> IO ()
runTensorValue (TScalar _ _) = return ()
runTensorValue (TTensor _ _ ref) = do
    value <- readIORef ref
    case value of
      TBuild build -> do
        vec <- TF.runSession $ TF.run build
        writeIORef ref $ TVector vec
      TVector _ -> return ()
      TNil -> return ()
