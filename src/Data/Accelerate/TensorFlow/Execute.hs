
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}



{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use record patterns" #-}
{-# LANGUAGE BangPatterns #-}


module Data.Accelerate.TensorFlow.Execute where
import Data.Array.Accelerate.AST.Schedule.Sequential
import Data.Accelerate.TensorFlow.Kernel
import Data.Array.Accelerate hiding (size, Vector, Exp)
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Text.Prettyprint.Doc (viaShow)
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.AST.Kernel
import Data.Array.Accelerate.Interpreter hiding (executeKernelFun, Right, Left)

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
import Data.Array.Accelerate.Trafo.Var

-- why does it not fully evaluate writeIOref, undefined, etc?

type TensorEnv = Env TensorValue

data TensorV a where
  TBuild :: TF.Tensor TF.Build a -> TensorV a
  TVector :: S.Vector a -> TensorV a
  TPlaceholder :: TF.Tensor TF.Value a -> TensorV a

data TensorValue a where
  TScalar :: a -> TensorValue a -- this should be a scalar?
  TTensor :: (Show a, S.Storable a) => IORef (TensorV a) -> TensorValue (Buffer a)

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
executeSequentialSchedule' env (Exec _ fun args) = executeKernelFun env fun args

executeSequentialSchedule' env (Return vars) = return $ mapTupR (\(Var _ idx) -> prj' idx env) vars

executeSequentialSchedule' env (Compute expr) = return (TupRsingle (TScalar (evalExp expr (EvalArrayInstr $ evalArrayInstr env))))

executeSequentialSchedule' env (Alet lhs _ sched sched') = do
  rhs <- executeSequentialSchedule' env sched
  let env' = push env (lhs, rhs)
  executeSequentialSchedule' env' sched'

executeSequentialSchedule' env (Alloc shR st vars)
  | TensorTypeDict <- tensorTypeDict st
  , VectorTypeDict <- vectorTypeDict st
  = do
  sh <- liftIO $ getShape env shR vars
  x <- TF.placeholder sh
  ref <- liftIO $ newIORef $ TPlaceholder x
  return $ TupRsingle $ TTensor ref

executeSequentialSchedule' _ (Use st n buffer)
  | TensorTypeDict <- tensorTypeDict st
  , VectorTypeDict <- vectorTypeDict st
  = do
  x <- fromBuffer dim1 st ((), n) buffer
  ref <- liftIO $ newIORef $ TVector x
  return $ TupRsingle $ TTensor ref

executeSequentialSchedule' _ (Unit var) = undefined

executeSequentialSchedule' _ (Acond var ss ss') = undefined

executeSequentialSchedule' _ (Awhile tr ssf ssf' tr') = undefined

executeKernelFun :: TensorEnv env -> KernelFun TensorKernel args -> SArgs env args -> TF.Session (TensorValues ())
executeKernelFun = executeKernelFun' Empty

executeKernelFun' :: TensorEnv env' -> TensorEnv env -> OpenKernelFun TensorKernel env' args -> SArgs env args -> TF.Session (TensorValues ())
executeKernelFun' env' env (KernelFunLam (KernelArgRscalar _) fun) ((SArgScalar (Var st idx)) :>: args)     =
  let value = prj' idx env in
    executeKernelFun' (push env' (LeftHandSideSingle st, TupRsingle value)) env fun args
executeKernelFun' env' env (KernelFunLam (KernelArgRbuffer _ _) fun) ((SArgBuffer _ (Var gr idx)) :>: args) =
  let value = prj' idx env in
    executeKernelFun' (push env' (LeftHandSideSingle gr, TupRsingle value)) env fun args
executeKernelFun' env' _ (KernelFunBody kernel) ArgsNil                                                     = executeKernel env' kernel

 -- de env moet empty zijn, help

executeKernel :: TensorEnv env -> TensorKernel env -> TF.Session (TensorValues ())
executeKernel env (TensorConstant shR t vars s (Var _ idx))
  | TensorTypeDict <- tensorTypeDict t
  , VectorTypeDict <- vectorTypeDict t
  = do
  let tensorVRef = case prj' idx env of
                TTensor ref -> ref
                _ -> error "impossible"
  tensorV <- liftIO $ readIORef tensorVRef
  case tensorV of
    TPlaceholder _ -> do -- TODO: investigate the use of placeholders
      (TF.Shape sh) <- liftIO $ getShape env shR vars
      let build = TF.fill (TF.vector sh) (TF.scalar s)
      liftIO $ writeIORef tensorVRef $ TBuild build
      return TupRunit
    _ -> error "placeholders only"

executeKernel env (TensorPrimFun shR fun shVars inVars outVars) =
  return TupRunit
executeKernel env (TensorId sr st tr var var')     =
  return TupRunit

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
      return (sh', case prj' idx env' of { TScalar sh -> sh })
    getShape' _ _ _ = error "impossible"

push :: TensorEnv env -> (LeftHandSide s t env env', TensorValues t) -> TensorEnv env'
push env (LeftHandSideWildcard _, _)            = env
push env (LeftHandSideSingle _  , TupRsingle a) = env `Push` a
push env (LeftHandSidePair l1 l2, TupRpair a b) = push env (l1, a) `push` (l2, b)
push _ _                                        = error "Tuple mismatch"

printTensorValue :: TensorValue a -> IO ()
printTensorValue (TScalar s) = putStr "TSCALAR" -- how do I print s?
printTensorValue (TTensor ref) = do
  x <- readIORef ref
  case x of
    TBuild _ -> putStr "BUILD"
    TVector vec -> putStr $ show vec
    TPlaceholder ten -> putStr "PLACEHOLDER"

printTensorValues :: TensorValues a -> IO ()
printTensorValues TupRunit        = putStr "()"
printTensorValues (TupRsingle t)  = printTensorValue t
printTensorValues (TupRpair t t') = do putStr "("
                                       printTensorValues t
                                       putStr ","
                                       printTensorValues t'
                                       putStr ")"

showScalar :: ScalarType e -> e -> String
showScalar (SingleScalarType t) e = showSingle t e
showScalar (VectorScalarType t) e = undefined

showSingle :: SingleType e -> e -> String
showSingle (NumSingleType t) = showNum t

showNum :: NumType e -> e -> String
showNum (IntegralNumType t) e = showIntegral t e
showNum (FloatingNumType t) e = showFloating t e

showIntegral :: IntegralType e -> e -> String
showIntegral TypeInt{}    e = show e
showIntegral TypeInt8{}   e = show e
showIntegral TypeInt16{}  e = show e
showIntegral TypeInt32{}  e = show e
showIntegral TypeInt64{}  e = show e
showIntegral TypeWord{}   e = show e
showIntegral TypeWord8{}  e = show e
showIntegral TypeWord16{} e = show e
showIntegral TypeWord32{} e = show e
showIntegral TypeWord64{} e = show e

showFloating :: FloatingType e -> e -> String
showFloating TypeHalf{}   e = show e
showFloating TypeFloat{}  e = show e
showFloating TypeDouble{} e = show e
