{-# LANGUAGE GADTs             #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE TypeOperators     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}


module Data.Accelerate.TensorFlow.Kernel where

import Data.Array.Accelerate.Lifetime
import Foreign
import Data.Array.Accelerate.AST.Kernel (NoKernelMetadata)
import Data.Text.Prettyprint.Doc
import Data.Array.Accelerate.Pretty.Exp
    ( prettyConst, primOperator, prettyLhs )
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Smart (typeR, undef)
import GHC.Conc (TVar(TVar))
import Data.Accelerate.TensorFlow.Operation
import Data.Array.Accelerate
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.Schedule.Uniform (BaseVar, BaseVars, fromGrounds, BaseR (BaseRground), Var (Var), mapArgs, GLeftHandSide)
import Data.Array.Accelerate.Type (ScalarType (SingleScalarType), NumType)
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.AST.Environment hiding (Val)
import Data.Array.Accelerate.AST.Partitioned
import Data.Array.Accelerate.Eval
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type (TupR(TupRsingle, TupRunit, TupRpair), Distributes (reprIsSingle), TypeR, mapTupR)
import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.Pretty.Print
import Data.Accelerate.TensorFlow.Type

data TensorKernel env where
  TensorConstant :: TensorDict TFAll s -> s -> BaseVar env (Buffer s) -> TensorKernel env
  TensorId :: TensorDict TFAll a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAdd :: TensorDict TFNum a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorMul :: TensorDict TFNum a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSub :: TensorDict TFNum a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorNeg :: TensorDict TFNeg a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAbs :: TensorDict TFAbs a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSign :: TensorDict TFSign a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env

  TensorTruncateDiv :: TensorDict TFNum a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env

instance NFData' TensorKernel where
  rnf' !_  = ()

newtype TensorFlowKernelMetadata f =
  TensorFlowKernelMetadata { kernelArgsSize :: Int }

instance IsKernel TensorKernel where
  type KernelOperation TensorKernel = TensorOp
  type KernelMetadata  TensorKernel = NoKernelMetadata

  compileKernel :: Env AccessGroundR env -> Cluster TensorOp args -> Args env args -> TensorKernel env
  compileKernel _ cluster clusterArgs
    | ClusterOperations _ lhs [ApplyOperation operation args] <- clusterOperations cluster clusterArgs
    , Just Refl <- wildcards lhs
    = compileOperation operation args
  compileKernel _ _ _ = error "impossible, did you use SequentialSchedule?"

wildcards :: LeftHandSide a e env env' -> Maybe (env :~: env')
wildcards (LeftHandSideWildcard _)    = Just Refl
wildcards (LeftHandSidePair lhs lhs')
 | Just Refl <- wildcards lhs
 , Just Refl <- wildcards lhs'        = Just Refl
wildcards _                           = Nothing

instance PrettyKernel TensorKernel where
  prettyKernel = PrettyKernelBody True $ \_ _ -> ""

compileOperation :: TensorOp args -> Args env args -> TensorKernel env
compileOperation (TConstant dict stIn s) (argIn :>: _)      = compileNullaryOperation1 dict stIn argIn (`TensorConstant` s)
compileOperation (TId dict stIn) (argIn :>: argOut :>: _)   = compileUnaryOperation1 dict stIn stIn argIn argOut TensorId
 
compileOperation (TAdd dict stIn) (argIn :>: argOut :>: _)  = compileBinaryOperation1 dict stIn stIn argIn argOut TensorAdd
compileOperation (TMul dict stIn) (argIn :>: argOut :>: _)  = compileBinaryOperation1 dict stIn stIn argIn argOut TensorMul
compileOperation (TSub dict stIn) (argIn :>: argOut :>: _)  = compileBinaryOperation1 dict stIn stIn argIn argOut TensorSub
compileOperation (TNeg dict stIn) (argIn :>: argOut :>: _)  = compileUnaryOperation1 dict stIn stIn argIn argOut TensorNeg
compileOperation (TAbs dict stIn) (argIn :>: argOut :>: _)  = compileUnaryOperation1 dict stIn stIn argIn argOut TensorAbs
compileOperation (TSign dict stIn) (argIn :>: argOut :>: _) = compileUnaryOperation1 dict stIn stIn argIn argOut TensorSign
compileOperation (TTruncateDiv dict stIn) (argIn :>: argOut :>: _) = compileBinaryOperation1 dict stIn stIn argIn argOut TensorTruncateDiv
compileOperation _ _ = undefined

compileNullaryOperation1 :: forall types a sh env. TensorDict types a -> ScalarType a -> Arg env (Out sh a) -> (TensorDict types a -> BaseVar env (Buffer a) -> TensorKernel env) -> TensorKernel env
compileNullaryOperation1 TensorDict stOut (ArgArray _ (ArrayR _ a) _ gvbOut) kernel
 | Refl <- reprIsSingle @ScalarType @a @Buffer stOut
 = kernel TensorDict (groundToBase a gvbOut)

compileBinaryOperation1 :: forall types a b sh env. TensorDict types a -> ScalarType a -> ScalarType b -> Arg env (In sh (a, a)) -> Arg env (Out sh b) -> (TensorDict types a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer b) -> TensorKernel env) -> TensorKernel env
compileBinaryOperation1 TensorDict stIn stOut (ArgArray _ (ArrayR _ (TupRpair a a')) _ (TupRpair gvbIn gvbIn')) (ArgArray _ (ArrayR _ b) _ gvbOut) kernel
 | Refl <- reprIsSingle @ScalarType @a @Buffer stIn
 , Refl <- reprIsSingle @ScalarType @b @Buffer stOut
 = kernel TensorDict (groundToBase a gvbIn) (groundToBase a' gvbIn') (groundToBase b gvbOut)
compileBinaryOperation1 _ _ _ _ _ _ = error "impossible"

compileUnaryOperation1 :: forall types a b sh env. TensorDict types a -> ScalarType a -> ScalarType b -> Arg env (In sh a) -> Arg env (Out sh b) -> (TensorDict types a -> BaseVar env (Buffer a) -> BaseVar env (Buffer b) -> TensorKernel env) -> TensorKernel env
compileUnaryOperation1 TensorDict stIn stOut (ArgArray _ (ArrayR _ a) _ gvbIn) (ArgArray _ (ArrayR _ b) _ gvbOut) kernel
  | Refl <- reprIsSingle @ScalarType @a @Buffer stIn
  , Refl <- reprIsSingle @ScalarType @b @Buffer stOut
  = kernel TensorDict (groundToBase a gvbIn) (groundToBase b gvbOut)

groundToBase :: TypeR a -> GroundVars env (Buffer a) -> BaseVar env (Buffer a)
groundToBase _ (TupRsingle (Var groundR idx)) = Var (BaseRground groundR) idx

groundsToBase :: TypeR a -> GroundVars env (Buffers a) -> BaseVars env (Buffers a)
groundsToBase _ TupRunit                              = TupRunit
groundsToBase t@(TupRsingle (st :: ScalarType e)) gvb
  | Refl <- reprIsSingle @ScalarType @e @Buffer st    = TupRsingle (groundToBase t gvb)
groundsToBase (TupRpair t1 t2) (TupRpair gvb1 gvb2)   = TupRpair (groundsToBase t1 gvb1) (groundsToBase t2 gvb2)
groundsToBase _ _                                     = error "impossible"