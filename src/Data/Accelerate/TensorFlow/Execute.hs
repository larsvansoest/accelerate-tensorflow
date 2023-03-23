{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use record patterns" #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE InstanceSigs #-}
module Data.Accelerate.TensorFlow.Execute where
import Data.Array.Accelerate.AST.Schedule.Sequential
import Data.Accelerate.TensorFlow.Kernel
import Data.Array.Accelerate hiding (Vector, Exp)
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Text.Prettyprint.Doc (viaShow)
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.AST.Kernel
import Data.Array.Accelerate.Interpreter hiding (Right, Left)

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
import Data.Accelerate.TensorFlow.Tensor hiding (toBuffer)
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Trafo.Operation.Substitution
import qualified Data.Vector.Storable as S
import Control.Concurrent.MVar
import Data.Accelerate.TensorFlow.Type


type TensorEnv = Env TensorValue

data TensorValue a where
  TVScalar :: a -> TensorValue a
  TVTensor :: (Show a, S.Storable a) => IORef (Either (TF.Tensor TF.Build a) (S.Vector a)) -> TensorValue (Buffer a)
  TVPlaceholder :: IORef (TF.Tensor TF.Value a) -> TensorValue (Buffer a)

type TensorValues = TupR TensorValue

type family TensorValuesIOFun t where
  TensorValuesIOFun (a -> b) = TensorValues a -> TensorValuesIOFun b
  TensorValuesIOFun t        = IO t

executeSequentialSchedule :: TensorEnv env -> SequentialSchedule TensorKernel env t -> TensorValuesIOFun t
executeSequentialSchedule env (SequentialLam lhs sched) = \values -> executeSequentialSchedule (push env (lhs, values)) sched
executeSequentialSchedule env (SequentialBody sched) = \_ -> do
  x <- TF.runSession $ executeSequentialSchedule' env sched
  printTensorValues x

executeSequentialSchedule' :: TensorEnv env -> SeqSchedule TensorKernel env t -> TF.Session (TensorValues t)
executeSequentialSchedule' env (Exec m fun args) = executeKernel env m fun args

executeSequentialSchedule' env (Return vars) = return $ mapTupR (\(Var _ idx) -> prj' idx env) vars

executeSequentialSchedule' env (Compute expr) = return (TupRsingle (TVScalar (evalExp expr (EvalArrayInstr $ evalArrayInstr env))))

executeSequentialSchedule' env (Alet lhs _ sched sched') = do
  rhs <- executeSequentialSchedule' env sched
  let env' = push env (lhs, rhs)
  executeSequentialSchedule' env' sched'

executeSequentialSchedule' env (Alloc shR st vars) 
  | TensorTypeDict <- tensorTypeDict st 
  = do
  sh <- liftIO $ getShape env shR vars 
  x <- TF.placeholder sh
  ref <- liftIO $ newIORef x
  return $ TupRsingle $ TVPlaceholder ref

executeSequentialSchedule' _ (Use st n buffer)
  | TensorTypeDict <- tensorTypeDict st 
  , VectorTypeDict <- vectorTypeDict st
  = do
  x <- fromBuffer dim1 st ((), n) buffer
  ref <- liftIO $ newIORef $ Right x
  return $ TupRsingle $ TVTensor ref

executeSequentialSchedule' _ (Unit var) = undefined

executeSequentialSchedule' _ (Acond var ss ss') = undefined

executeSequentialSchedule' _ (Awhile tr ssf ssf' tr') = undefined

evalArrayInstr :: TensorEnv env -> ArrayInstr env (s -> t) -> s -> t
evalArrayInstr env (Index (Var _ idx))     = undefined
evalArrayInstr env (Parameter (Var _ idx)) = undefined

getShape :: TensorEnv env -> ShapeR sh -> ExpVars env sh -> IO TF.Shape
getShape env shR vars = do
  sh <- liftIO $ getShape' env shR vars
  return $ toTFShape shR sh
  where
    getShape' :: TensorEnv env -> ShapeR sh -> ExpVars env sh -> IO sh
    getShape' _ ShapeRz _                                                      = return ()
    getShape' env' (ShapeRsnoc shR') (TupRpair vars1 (TupRsingle (Var _ idx))) = do
      sh' <- getShape' env' shR' vars1
      return (sh', case prj' idx env' of { TVScalar sh -> sh })
    getShape' _ _ _ = error "impossible"

executeKernel :: TensorEnv env -> NoKernelMetadata f -> KernelFun TensorKernel args -> SArgs env args -> TF.Session (TensorValues t)
executeKernel env m (KernelFunLam z kernel) args = undefined
executeKernel env m (KernelFunBody kernel) args = undefined

toBuffer :: TensorValue (Buffer t) -> IO (Buffer t)
toBuffer (TVTensor ref) = undefined
toBuffer _ = undefined

push :: TensorEnv env -> (LeftHandSide s t env env', TensorValues t) -> TensorEnv env'
push env (LeftHandSideWildcard _, _)            = env
push env (LeftHandSideSingle _  , TupRsingle a) = env `Push` a
push env (LeftHandSidePair l1 l2, TupRpair a b) = push env (l1, a) `push` (l2, b)
push _ _                                        = error "Tuple mismatch"

printTensorValue :: TensorValue a -> IO ()
printTensorValue (TVScalar s) = do putStr "TVScalar"
printTensorValue (TVTensor ref) = do 
  x <- readIORef ref
  case x of
    Left ten -> putStr "Build"
    Right vec -> putStr $ show vec

printTensorValues :: TensorValues a -> IO ()
printTensorValues TupRunit        = do putStr "()"
printTensorValues (TupRsingle t)  = do printTensorValue t
printTensorValues (TupRpair t t') = do putStr "("
                                       printTensorValues t
                                       putStr ","
                                       printTensorValues t'
                                       putStr ")"
  
